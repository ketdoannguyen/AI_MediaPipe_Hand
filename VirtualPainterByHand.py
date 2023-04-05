import cv2
import time
import numpy as np
import HandTrackingModule as htm

SIZE_DRAW = 15
SIZE_ERASER = 30

def statusFingers(status, finger, frame, x1, y1, x2, y2):
    drawColor = (255, 255, 255)
    if status == 0:
        drawColor = (255, 255, 255)
    elif status == 1:
        drawColor = (255, 153, 51)
    elif status == 2:
        drawColor = (10, 102, 10)
    elif status == 3:
        drawColor = (10, 10, 204)
    if finger == 1:
        if status == 0:
            cv2.circle(frame, (x1, y1), SIZE_ERASER, drawColor, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x1, y1), SIZE_DRAW, drawColor, -1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (x1, y1), SIZE_DRAW, drawColor, -1, cv2.LINE_AA)
        cv2.circle(frame, (x2, y2), SIZE_DRAW, drawColor, -1, cv2.LINE_AA)
    return drawColor

def drawOnImage(frame, img_blank):
    mask = img_blank > 0
    # Use the boolean mask to replace values in frame with values in img_blank
    frame[mask] = img_blank[mask]
    return frame

def main():
    pTime = 0
    cTime = 0

    capture = cv2.VideoCapture(0)
    height, weight = 900, 1600
    capture.set(3, 1920)
    capture.set(4, 1080)

    img_bg_head = cv2.imread("image/bg_head.png")
    h_head,w_head = img_bg_head.shape[:2]
    img_blank = np.zeros((height, weight, 3), dtype=np.uint8)

    detector = htm.HandDetector(maxNumHands=1)
    status = 0
    pre_point_draw = ()
    while capture.isOpened():
        success, frame = capture.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (weight, height))

        if success is not None:
            frame = detector.findHands(frame)
            listLmsAll = detector.findPosition(frame, maxNumHand=1)

            frame[0:h_head, 0:weight, :] = cv2.addWeighted(img_bg_head, 0.8, frame[0:h_head, 0:weight, :], 0.2, 0)
            if len(listLmsAll) > 0:
                listLms = listLmsAll[0]
                finger, _ = detector.fingersUp(listLms)
                x1, y1 = listLms[8][1], listLms[8][2]
                x2, y2 = listLms[12][1], listLms[12][2]

                if finger[1] and not finger[2]:
                    drawColor = statusFingers(status, 1, frame, x1, y1, x2, y2)
                    if len(pre_point_draw) > 0:
                        if status == 0:
                            cv2.line(img_blank, pre_point_draw, (x1, y1), (0, 0, 0), SIZE_ERASER*2, cv2.LINE_AA)
                        else:
                            cv2.line(img_blank, pre_point_draw, (x1, y1), drawColor, SIZE_DRAW, cv2.LINE_AA)
                    pre_point_draw = (x1, y1)
                elif finger[1] and finger[2]:
                    if abs(x1 - x2) < 70:
                        if 0 < y1 < h_head:
                            if 450 < x1 < 630:
                                status = 1
                            elif 750 < x1 < 930:
                                status = 2
                            elif 1040 < x1 < 1225:
                                status = 3
                            elif 1340 < x1 < 1500:
                                status = 0
                        statusFingers(status, 2, frame, x1, y1, x2, y2)
                        pre_point_draw = ()
                elif finger == [0, 0, 0, 0, 1]:
                    pre_point_draw = ()
                    img_blank = np.zeros((height, weight, 3), dtype=np.uint8)
                else:
                    pre_point_draw = ()

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(frame, "FPS:" + str(int(fps)), (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2,
                        cv2.LINE_AA)
            frame = drawOnImage(frame, img_blank)
            cv2.imshow("Virtual Painter", frame)
            if cv2.waitKey(8) & 0xFF == 113:
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
