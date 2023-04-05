import cv2
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0

capture = cv2.VideoCapture(0)
height, weight = 1080, 960
capture.set(3, height)
capture.set(4, weight)

detector = htm.HandDetector(maxNumHands=5)
while capture.isOpened():
    success, frame = capture.read()
    frame = cv2.flip(frame, 1)
    if success is not None:
        frame = detector.findHands(frame)
        listLmsAll = detector.findPosition(frame, maxNumHand=2, draw=True)
        if len(listLmsAll) > 0:
            for numHands, listLmsEach, handType in listLmsAll:
                max_h, min_h, max_w, min_w = 0, height, 0, weight
                for id, cx, cy in listLmsEach:
                    max_h = max(cy, max_h)
                    min_h = min(cy, min_h)
                    max_w = max(cx, max_w)
                    min_w = min(cx, min_w)
                cv2.rectangle(frame, (min_w, min_h), (max_w, max_h), (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, handType, (min_w, min_h - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2,
                            cv2.LINE_AA)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, "FPS:" + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Tracking Hand", frame)
        if cv2.waitKey(1) & 0xFF == 113:
            break
    else:
        break
capture.release()
cv2.destroyAllWindows()
