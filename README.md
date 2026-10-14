# AI Attendance System

An AI-powered attendance system built using Python, TensorFlow, and OpenCV.

## Features
- AI-based student recognition using a trained model.
- Attendance marking after sustained recognition.
- Local attendance storage using SQLite.
- Telegram notifications with automatic retry support.
- CSV attendance export.

## Technologies Used
- Python
- TensorFlow
- OpenCV
- NumPy
- SQLite
- Telegram Bot API

## Project Files
- `attendance_telegram.py` — Main application.
- `labels.txt` — Model class labels.
- `model.savedmodel/` — Trained AI model.
- `webcam_test.py` — Webcam testing script.

## How to Run
1. Install the required Python libraries from `requirements.txt`.
2. Configure the Telegram bot environment variables.
3. Run `attendance_telegram.py`.

## Notes
Attendance records and database files are stored locally. Telegram credentials should be configured through environment variables and must never be uploaded to GitHub.
