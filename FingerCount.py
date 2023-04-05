import cv2
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0

capture = cv2.VideoCapture(0)
height, weight = 720, 1280
capture.set(3, weight)
capture.set(4, height)

detector = htm.HandDetector(maxNumHands=5)
while capture.isOpened():
    success, frame = capture.read()
    frame = cv2.flip(frame, 1)
    if success is not None:
        frame = detector.findHands(frame)
        listLmsAll = detector.findPosition(frame, maxNumHand=2, draw=False)
        if len(listLmsAll) > 0:
            fingerCount = 0
            for numHands, listLmsEach, handType in listLmsAll:
                fingerList, idxTipsTrue = detector.fingersUp(listLmsEach, handType)
                fingerCount += sum(fingerList)
                for i in idxTipsTrue:
                    cx, cy = listLmsEach[i][1], listLmsEach[i][2]
                    cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, str(int(fingerCount)), (20, 710), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 255), 5,
                        cv2.LINE_AA)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, "FPS:" + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Finger Count", frame)
        if cv2.waitKey(1) & 0xFF == 113:
            break
    else:
        break
capture.release()
cv2.destroyAllWindows()
