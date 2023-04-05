import cv2
import time
import numpy as np
import HandTrackingModule as htm

COLOR_CHAR_KB = (223, 53, 57)
COLOR_RECT_KB = (255, 100, 255)
COLOR_FINGER_TIP = (41, 0, 223)


def drawRectangleAndAlpha(image, alpha, left, top, right, bottom):
    cv2.rectangle(image, (left, top), (right, bottom), COLOR_RECT_KB, -1, cv2.LINE_AA)
    if len(alpha) == 1:
        cv2.putText(image, alpha, (left + 12, bottom - 12), cv2.FONT_HERSHEY_PLAIN, 8, COLOR_CHAR_KB, 10, cv2.LINE_AA)
    elif alpha == "space":
        cv2.putText(image, alpha, (left + 100, bottom - 25), cv2.FONT_HERSHEY_PLAIN, 6, COLOR_CHAR_KB, 8, cv2.LINE_AA)
    else:
        cv2.putText(image, alpha, (left + 2, bottom - 15), cv2.FONT_HERSHEY_PLAIN, 6, COLOR_CHAR_KB, 10, cv2.LINE_AA)
    return image


def merge2Img(frame, img_blank):
    mask = img_blank > 0
    # Use the boolean mask to replace values in frame with values in img_blank
    frame[mask] = img_blank[mask] * 0.6 + frame[mask] * 0.4
    return frame


def charAndPosition():
    listCharAndPosition = []
    x0 = 30
    y0 = 70
    w = 105
    h = 105
    for i in range(10):
        listCharAndPosition.append((str(i), x0, y0, x0 + w, y0 + h))
        x0 = x0 + w + 25
    listCharAndPosition.append((str("del"), x0, y0, x0 + 160, y0 + h))
    x0 = 30
    y0 = y0 + h + 30
    for i in range(65, 76):
        listCharAndPosition.append((chr(i), x0, y0, x0 + w, y0 + h))
        x0 = x0 + w + 25
    x0 = 30
    y0 = y0 + h + 30
    listCharAndPosition.append((str("Cap"), x0, y0, x0 + 180, y0 + h))
    x0 = x0 + 205
    for i in range(76, 86):
        listCharAndPosition.append((chr(i), x0, y0, x0 + w, y0 + h))
        x0 = x0 + w + 25
    x0 = 30
    y0 = y0 + h + 30
    for i in range(86, 90):
        listCharAndPosition.append((chr(i), x0, y0, x0 + w, y0 + h))
        x0 = x0 + w + 25
    listCharAndPosition.append((str("space"), x0, y0, x0 + 495, y0 + h))
    x0 = x0 + 510
    listCharAndPosition.append((chr(90), x0, y0, x0 + w, y0 + h))
    x0 = x0 + w + 25
    listCharAndPosition.append((str(","), x0, y0, x0 + w, y0 + h))
    x0 = x0 + w + 25
    listCharAndPosition.append((str("."), x0, y0, x0 + w, y0 + h))
    return listCharAndPosition


def main():
    pTime = 0
    cTime = 0

    capture = cv2.VideoCapture(0)
    capture.set(3, 1920)
    capture.set(4, 1080)

    height, weight = 900, 1600

    img_blank = np.zeros((height, weight, 3), dtype=np.uint8)
    listAlphaAndPosition = charAndPosition()
    for alpha, left, top, right, bottom in listAlphaAndPosition:
        img_blank = drawRectangleAndAlpha(img_blank, alpha, left, top, right, bottom)
    cv2.circle(img_blank, (190, 360), 10, (255, 255, 255), -1, cv2.LINE_AA)

    pre_alpha = ""
    text = ""
    status_capslock = True

    detector = htm.HandDetector(maxNumHands=1)
    while capture.isOpened():
        success, frame = capture.read()
        frame = cv2.resize(frame, (weight, height))
        frame = cv2.flip(frame, 1)
        if success is not None:
            frame = detector.findHands(frame)
            listLmsAll = detector.findPosition(frame)
            if len(listLmsAll) > 0:
                listLms = listLmsAll[0]
                finger, _ = detector.fingersUp(listLms)
                x1, y1 = listLms[8][1], listLms[8][2]
                x2, y2 = listLms[12][1], listLms[12][2]

                if finger[1] and not finger[2]:
                    cv2.circle(frame, (x1, y1), 10, COLOR_FINGER_TIP, -1, cv2.LINE_AA)
                    pre_alpha = ""
                elif finger[1] and finger[2]:
                    cv2.circle(frame, (x1, y1), 10, COLOR_FINGER_TIP, -1, cv2.LINE_AA)
                    cv2.circle(frame, (x2, y2), 10, COLOR_FINGER_TIP, -1, cv2.LINE_AA)
                    for alpha, left, top, right, bottom in listAlphaAndPosition:
                        if left <= x1 <= right and top <= y1 <= bottom:
                            if alpha != pre_alpha:
                                if alpha == "del":
                                    text = text[:-1]
                                elif alpha == "space":
                                    text += " "
                                elif alpha == "Cap":
                                    if status_capslock:
                                        cv2.circle(img_blank, (right - 20, top + 20), 10, COLOR_RECT_KB, -1,
                                                   cv2.LINE_AA)
                                        status_capslock = False
                                    else:
                                        cv2.circle(img_blank, (right - 20, top + 20), 10, (255, 255, 255), -1,
                                                   cv2.LINE_AA)
                                        status_capslock = True
                                else:
                                    if not status_capslock and alpha.isalpha():
                                        text += alpha.lower()
                                    else:
                                        text += alpha

                            pre_alpha = alpha
                            break
                elif finger == [0, 0, 0, 0, 1]:
                    text = ""
            cv2.rectangle(img_blank, (30, 600), (1425, 720), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.rectangle(img_blank, (30, 600), (1425, 720), COLOR_RECT_KB, -1, cv2.LINE_AA)
            cv2.putText(img_blank, text[-18:], (40, 710), cv2.FONT_HERSHEY_PLAIN, 8, (128, 128, 1), 10, cv2.LINE_AA)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, "FPS:" + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2,
                        cv2.LINE_AA)

            frame = merge2Img(frame, img_blank)
            cv2.imshow("Simulate Keyboard", frame)
            if cv2.waitKey(8) & 0xFF == 113:
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()