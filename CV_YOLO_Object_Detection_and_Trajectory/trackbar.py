import cv2
import numpy as np


# cap=cv2.VideoCapture("OpenCV\\Res\\ball_for_tracking.mov")
img = cv2.imread(r"Res\bit1.png")
img = cv2.resize(img, (700, 500))


def Track(x):
    pass


cv2.namedWindow("Color detectors")
cv2.resizeWindow("Color detectors", 500, 500)
cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, Track)
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, Track)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, Track)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, Track)
cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255, Track)
cv2.createTrackbar("Lower Value", "Color detectors", 49, 255, Track)


# imgs=cv2.resize(img,(500,500))
while True:
    # success,img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")

    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")

    print(
        f"upper\n",
        {u_hue, u_saturation, u_value},
        "\n lower",
        {l_hue, l_saturation, l_value},
    )

    Upper_hsv = np.array([u_hue, u_saturation, u_value])
    Lower_hsv = np.array([l_hue, l_saturation, l_value])
    mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    res = cv2.bitwise_and(img, img, mask=mask)
    res = cv2.medianBlur(res, 5)

    cv2.imshow("video", img)
    cv2.imshow("res", res)
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

cv2.destroyAllWindows