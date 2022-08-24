import cv2
import numpy as np
import time
import os
import mediapipe as mp
import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 50
folderPath = "Header"
# myList = os.listdir(folderPath)

myList = ['1.png', '2.png', '3.png', '4.png']
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)

wCam, hCam = 1080, 820
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((820, 1080, 3), np.uint8)
while True:

    # Import image
    success, img = cap.read()
    img = cv2.resize(img, (1080, 820))
    img = cv2.flip(img, 1)

    # Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPostition(img, draw=False)
    if len(lmList) != 0:
        # Tip of index and middle finger
        _, x1, y1 = lmList[8]
        _, x2, y2 = lmList[12]

        # Checking which fingers are up
        fingers = detector.fingersUP()

        # If Selection Mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection")
            if y1 < 131:
                if 275 < x1 < 350:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 450 < x1 < 525:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 625 < x1 < 700:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 800 < x1 < 1080:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # If Drawing Mode - Index Figure is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:131, 0:1080] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)

    cv2.waitKey(1)
