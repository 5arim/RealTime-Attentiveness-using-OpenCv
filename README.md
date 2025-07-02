🧠 Class Pulse – Real-Time Alertness Detection using OpenCV & MediaPipe
Class Pulse is a Final Year Project (FYP) designed to monitor student attentiveness in real-time using computer vision. Built with OpenCV and MediaPipe, this system detects face presence, head orientation, and eye gaze direction to calculate an overall alertness score for each session. It provides a non-invasive, automated way to assess student engagement during online or physical classes.

🎯 Key Features
👁️ Real-Time Face Detection using MediaPipe FaceMesh

📏 Head Orientation Tracking to detect attention shifts

👀 Gaze Direction Estimation (left, right, center)

📊 Alertness Scoring System based on visual behavior

📝 Session Logs for performance review and feedback

🧰 Technologies Used
Python 3

OpenCV

MediaPipe

NumPy, Pandas (for data handling)

Matplotlib / Seaborn (for optional visualization)

(Optional) Flask or Streamlit for web-based interface

🚀 How It Works
Face Mesh is initialized using MediaPipe to track facial landmarks

Head orientation is determined by landmark geometry

Eye region analysis identifies gaze direction

Each second is scored as attentive or distracted

At the end of the session, a summary report is generated

💡 Use Cases
Monitoring student attentiveness during lectures

Detecting drowsiness or distraction in workplace environments

Research in education, psychology, and human-computer interaction

📂 Folder Structure
📁 class-pulse/
├── 📁 src/              # Python scripts for detection logic
├── 📁 data/             # Session recordings or logs
├── 📁 models/           # Pretrained face landmarks (if needed)
├── app.py              # Main executable script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
