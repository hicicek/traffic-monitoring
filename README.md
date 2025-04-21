# 🚦 Traffic Monitoring using YOLOv8
A real-time vehicle tracking project that analyzes traffic flow in a 3-way roundabout using computer vision.

---

## 🎥 Demo

Check out the 14-second demo video to see it in action!  
👉 [Watch Demo Video](./Demo.mp4)

---

## 🧠 What It Does

This project detects and tracks vehicles as they pass through a roundabout. Here's how it works:

- The roundabout has **3 entry roads**, each color-coded (Red, Yellow, Green).
- Each detected vehicle is **tagged based on where it exits** the roundabout.
- When the vehicle enters, the system:
  - **Identifies the exit path**
  - **Increases a count** for that enter
- It uses bounding boxes, entry tagging, and custom logic to make everything work in real-time.

It’s simple, effective, and fun to watch!

---

## 🛠️ Technologies Used

- 🧠 [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- 🎥 OpenCV - Video processing
- 📊 NumPy - Matrix calculations
- 📦 Supervision (by Roboflow) - Helpful annotation tools
- 🔁 tqdm - For progress bars
