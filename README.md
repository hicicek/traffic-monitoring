# 🚦 Traffic Monitoring using YOLOv8
A real-time vehicle tracking project that analyzes traffic flow in a 3-way roundabout using computer vision.

🎥 [Click here to watch the 14s demo video](./Demo.mp4)

This project uses YOLOv8 to detect and track vehicles as they pass through a 3-way roundabout. Each road is assigned a color (red, yellow, or green), and vehicles are tagged based on where they exit. When a vehicle enters an area, the system increases the count for that road — helping visualize traffic patterns in real-time. It's designed to be simple and efficient, just plug in a traffic video and see the flow unfold!

Technologies Used:
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Supervision (Roboflow)
- NumPy
- tqdm

Future Ideas:
- Add vehicle speed estimation
- Improve lane-change detection
- Create a dashboard for live visual analytics
