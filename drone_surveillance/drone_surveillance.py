import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLOv8 model (pretrained on COCO dataset → detects "person")
model = YOLO("yolov8n.pt")  # small & fast, you can use yolov8m.pt or l.pt

# Open crowd video
video_path = "1.mp4"   # 🔁 Replace with your test video
cap = cv2.VideoCapture(video_path)

# Video writer (for saving output)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    # Store person centers for heatmap
    centers = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":  # detect only humans
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save center
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                centers.append((cx, cy))

    # Count detected people
    total_people = len(centers)
    cv2.putText(frame, f"AI detected {total_people} people", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Generate heatmap if people detected
    if centers:
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        for cx, cy in centers:
            cv2.circle(heatmap, (cx, cy), 50, 1, -1)  # blur circle per person

        heatmap = cv2.GaussianBlur(heatmap, (0, 0), 25)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

        # Blend with original frame
        frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    out.write(frame)
    cv2.imshow("Drone AI Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
