# createPolygons.py
import cv2
import pickle

img = cv2.imread('CarParkProject\carParkImg.png')  # Replace with an image from your video

# Empty list to hold positions
posList = []

# Mouse callback function
def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + 103 and y1 < y < y1 + 43:
                posList.pop(i)

    with open('polygons', 'wb') as f:
        pickle.dump(posList, f)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouseClick)

while True:
    img_copy = img.copy()
    for pos in posList:
        cv2.rectangle(img_copy, pos, (pos[0]+103, pos[1]+43), (255, 0, 255), 2)

    cv2.imshow("Image", img_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
