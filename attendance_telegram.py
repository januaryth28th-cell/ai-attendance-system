import cv2
import csv
import os
import time
import sqlite3
import threading
import json
import urllib.request
import urllib.error
import numpy as np
import tensorflow as tf

from pathlib import Path
from datetime import datetime
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model.savedmodel"
LABELS_PATH = BASE_DIR / "labels.txt"
DB_PATH = BASE_DIR / "attendance.db"

CONFIDENCE_THRESHOLD = 0.85
REQUIRED_SECONDS = 3.0
FLICKER_GRACE_SECONDS = 0.3
IMAGE_SIZE = (224, 224)

# Configure these as Windows environment variables:
# TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

SESSION_ID = str(uuid4())
STOP_EVENT = threading.Event()


def load_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            labels.append(parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else line)
    return labels


def is_nothing(label):
    return label.strip().lower() in {"nothing", "background", "none"}


def db_connect():
    return sqlite3.connect(DB_PATH, timeout=10)


def initialize_database():
    with db_connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                student TEXT NOT NULL,
                marked_at TEXT NOT NULL,
                confidence REAL NOT NULL,
                UNIQUE(session_id, student)
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS notification_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT NOT NULL UNIQUE,
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                sent_at TEXT,
                last_error TEXT
            )
        """)


def queue_notification(event_key, message):
    with db_connect() as db:
        db.execute("""
            INSERT OR IGNORE INTO notification_queue
            (event_key, message, status, attempts, next_attempt_at, created_at)
            VALUES (?, ?, 'pending', 0, 0, ?)
        """, (event_key, message, datetime.now().astimezone().isoformat(timespec="seconds")))


def mark_attendance(student, confidence):
    marked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with db_connect() as db:
        cursor = db.execute("""
            INSERT OR IGNORE INTO attendance
            (session_id, student, marked_at, confidence)
            VALUES (?, ?, ?, ?)
        """, (SESSION_ID, student, marked_at, confidence))
        inserted = cursor.rowcount == 1

    if inserted:
        message = (
            "✅ Attendance marked\n"
            f"Student: {student}\n"
            f"Time: {marked_at}\n"
            f"Confidence: {confidence * 100:.1f}%\n"
            f"Session: {SESSION_ID[:8]}"
        )
        queue_notification(f"{SESSION_ID}:{student}", message)
    return inserted


def export_session_csv():
    output_path = BASE_DIR / f"attendance_{SESSION_ID[:8]}.csv"
    with db_connect() as db:
        rows = db.execute("""
            SELECT student, marked_at, confidence
            FROM attendance WHERE session_id = ?
            ORDER BY marked_at
        """, (SESSION_ID,)).fetchall()
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["Student", "Marked At", "Confidence"])
        writer.writerows(rows)
    print(f"\nSession CSV saved to: {output_path}")


def telegram_send_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        body = json.loads(response.read().decode("utf-8"))
        if response.status != 200 or not body.get("ok", False):
            raise RuntimeError(f"Telegram API rejected message: {body}")
    return True


def notification_worker():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(
            "\nTelegram isn't configured yet. Notifications will remain queued.\n"
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, then restart."
        )

    while not STOP_EVENT.is_set():
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            STOP_EVENT.wait(5)
            continue

        with db_connect() as db:
            row = db.execute("""
                SELECT id, message, attempts FROM notification_queue
                WHERE status = 'pending' AND next_attempt_at <= ?
                ORDER BY id LIMIT 1
            """, (time.time(),)).fetchone()

        if row is None:
            STOP_EVENT.wait(2)
            continue

        queue_id, message, attempts = row
        try:
            telegram_send_message(message)
            with db_connect() as db:
                db.execute("""
                    UPDATE notification_queue
                    SET status = 'sent', sent_at = ?, last_error = NULL
                    WHERE id = ?
                """, (datetime.now().astimezone().isoformat(timespec="seconds"), queue_id))
            print("\nTelegram notification sent.")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            new_attempts = attempts + 1
            delay = min(5 * (2 ** min(new_attempts - 1, 6)), 300)
            with db_connect() as db:
                db.execute("""
                    UPDATE notification_queue
                    SET attempts = ?, next_attempt_at = ?, last_error = ?
                    WHERE id = ?
                """, (new_attempts, time.time() + delay, str(exc)[:500], queue_id))
            print(f"\nTelegram send failed; retrying in {delay}s. Details: {exc}")
            STOP_EVENT.wait(1)


def main():
    if not MODEL_DIR.exists() or not LABELS_PATH.exists():
        print("ERROR: model.savedmodel or labels.txt is missing.")
        return

    labels = load_labels(LABELS_PATH)
    print("Loading TensorFlow model...")
    model = tf.saved_model.load(str(MODEL_DIR))
    infer = model.signatures["serving_default"]

    input_key = list(infer.structured_input_signature[1].keys())[0]
    output_key = list(infer.structured_outputs.keys())[0]
    input_shape = infer.structured_input_signature[1][input_key].shape
    output_shape = infer.structured_outputs[output_key].shape

    if tuple(input_shape[-3:]) != (224, 224, 3):
        print(f"ERROR: Expected model input (224, 224, 3), got {input_shape[-3:]}")
        return
    if output_shape[-1] != len(labels):
        print(f"ERROR: Model has {output_shape[-1]} outputs, labels.txt has {len(labels)} labels.")
        return

    initialize_database()
    worker = threading.Thread(target=notification_worker, daemon=True)
    worker.start()

    print("Session started:", SESSION_ID)
    print("Students:", [name for name in labels if not is_nothing(name)])
    print("Press Q to finish. SQLite records and queued alerts persist across restarts.")

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open webcam. Close other camera apps and retry.")
        STOP_EVENT.set()
        worker.join(timeout=2)
        return

    candidate = None
    qualified_seconds = 0.0
    last_good_time = None
    flicker_started = None
    candidate_confidence = 0.0
    status = "Show one student to the camera"

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("\nERROR: Could not read webcam frame.")
                break

            now = time.monotonic()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, IMAGE_SIZE)
            image = resized.astype(np.float32)
            image = np.expand_dims((image / 127.5) - 1.0, axis=0)

            result = infer(**{input_key: tf.convert_to_tensor(image)})
            scores = result[output_key].numpy()[0]
            index = int(np.argmax(scores))
            label = labels[index]
            confidence = float(scores[index])
            valid = not is_nothing(label) and confidence >= CONFIDENCE_THRESHOLD

            if not valid:
                if candidate is not None:
                    if flicker_started is None:
                        flicker_started = now
                    elif now - flicker_started > FLICKER_GRACE_SECONDS:
                        candidate = None
                        qualified_seconds = 0.0
                        last_good_time = None
                        flicker_started = None
                status = "Waiting for stable recognition"

            elif candidate is None:
                candidate = label
                qualified_seconds = 0.0
                last_good_time = now
                flicker_started = None
                candidate_confidence = confidence
                status = f"Recognizing {label}"

            elif candidate != label:
                candidate = label
                qualified_seconds = 0.0
                last_good_time = now
                flicker_started = None
                candidate_confidence = confidence
                status = f"Recognizing {label}"

            else:
                if flicker_started is not None:
                    gap = now - flicker_started
                    if gap > FLICKER_GRACE_SECONDS:
                        qualified_seconds = 0.0
                    flicker_started = None
                    last_good_time = now
                else:
                    if last_good_time is not None:
                        qualified_seconds += max(0.0, now - last_good_time)
                    last_good_time = now
                candidate_confidence = confidence
                status = f"Recognizing {candidate}"

            if candidate is not None and qualified_seconds >= REQUIRED_SECONDS:
                if mark_attendance(candidate, candidate_confidence):
                    status = f"MARKED PRESENT: {candidate}"
                    print(f"\nPRESENT: {candidate} — notification queued.")
                else:
                    status = f"Already marked: {candidate}"

            cv2.putText(
                frame, f"Prediction: {label} ({confidence * 100:.1f}%)",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if valid else (0, 165, 255), 2
            )
            if candidate is not None:
                cv2.putText(
                    frame, f"Timer: {qualified_seconds:.1f}/{REQUIRED_SECONDS:.1f}s",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2
                )
            cv2.putText(
                frame, status, (12, 92), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )
            cv2.imshow("AI Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        STOP_EVENT.set()
        worker.join(timeout=10)
        export_session_csv()
        print("Session ended. Unsent notifications remain in attendance.db for the next launch.")


if __name__ == "__main__":
    main()
