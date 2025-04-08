import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model (make sure yolov8n.pt is downloaded)
model = YOLO("yolov8n.pt")

# IP Webcam URL or video stream
url = 'http://192.168.1.100:8080/video'  # Replace with your webcam URL
cap = cv2.VideoCapture(url)

# ----------------- Helper: Filter Crosswalk Lines -----------------
def filter_crosswalk_lines(lines):
    if lines is None:
        return []

    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if -10 < angle < 10:  # near horizontal
            length = np.linalg.norm([x2 - x1, y2 - y1])
            if length > 50:
                horizontal_lines.append([x1, y1, x2, y2])

    if len(horizontal_lines) < 5:
        return []

    y_mids = [(line[1] + line[3]) / 2 for line in horizontal_lines]
    std_y = np.std(y_mids)
    if std_y > 25:
        return []

    return horizontal_lines

# ----------------- Main Loop -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame!")
        break

    frame = cv2.resize(frame, (640, 480))

    # ---------- CROSSWALK DETECTION ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    crosswalk_lines = filter_crosswalk_lines(lines)

    if crosswalk_lines:
        for x1, y1, x2, y2 in crosswalk_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        all_x = [x for line in crosswalk_lines for x in (line[0], line[2])]
        all_y = [y for line in crosswalk_lines for y in (line[1], line[3])]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        cv2.putText(frame, "Crosswalk Detected", (min_x, min_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        min_x, max_x, min_y, max_y = None, None, None, None

    # ---------- YOLO PERSON DETECTION ----------
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

                if crosswalk_lines and min_x and min_y:
                    # Check if person bounding box is within crosswalk region
                    if min_x < x1 < max_x and min_y < y1 < max_y:
                        cv2.putText(frame, "Person in Crosswalk", (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ---------- SHOW FRAME ----------
    cv2.imshow("Crosswalk & Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



