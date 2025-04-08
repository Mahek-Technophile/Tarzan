import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# IP Webcam URL (replace with your IP address)
url = 'http://192.168.1.100:8080/shot.jpg'  # Change to your IP Webcam URL

while True:
    # Read frame from IP Webcam
    img_resp = cv2.VideoCapture(url)
    ret, frame = img_resp.read()
    
    if not ret:
        print("Failed to grab frame!")
        break

    frame = cv2.resize(frame, (640, 480))

    # ----------- CROSSWALK DETECTION -----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

    crosswalk_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Consider near-horizontal lines only (for crosswalk)
            if -10 < angle < 10:
                crosswalk_lines.append(line[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw Crosswalk Region if lines detected
    if crosswalk_lines:
        all_x = [x for line in crosswalk_lines for x in (line[0], line[2])]
        all_y = [y for line in crosswalk_lines for y in (line[1], line[3])]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        cv2.putText(frame, "Crosswalk Detected", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # ----------- YOLO PERSON DETECTION -----------
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name == "person":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Optional: Check if person is in crosswalk region
                if crosswalk_lines:
                    if min_x < x1 < max_x and min_y < y1 < max_y:
                        cv2.putText(frame, "Person in Crosswalk", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ----------- SHOW FRAME -----------
    cv2.imshow("Crosswalk & Person Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
