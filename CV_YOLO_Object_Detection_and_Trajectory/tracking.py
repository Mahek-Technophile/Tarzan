import cv2
import numpy as np

video = cv2.VideoCapture(r"C:\Users\mahek\OneDrive\Desktop\CV Workshop\CV_Workshop-main\res\ball_for_tracking.mov")
cposl = []
while True:
    ret, img = video.read()
    if not ret:
        break
 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([0, 170, 90])
    upper_bound = np.array([4, 255, 255])

    mask = cv2.inRange(hsv,lower_bound,upper_bound)
    x, y, w, h = cv2.boundingRect(mask)  
    cv2.rectangle(img, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 1)

    

    center =( ( x+w ), (y + h))

    cposl.append(center)
    for pos in cposl:
        cv2.circle(img, pos,2, (255,255,255), -1)

    cv2.imshow("video", img)
    #cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break
 


cv2.destroyAllWindows()