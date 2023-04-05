import cv2
import time
import HandTrackingModule as htm
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def distance2Point(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def main():
    pTime = 0
    cTime = 0

    capture = cv2.VideoCapture(0)
    height, weight = 1080, 960
    capture.set(3, height)
    capture.set(4, weight)

    detector = htm.HandDetector(maxNumHands=1)

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    minVol = volume.GetVolumeRange()[0]
    maxVol = volume.GetVolumeRange()[1]
    while capture.isOpened():
        success, frame = capture.read()
        frame = cv2.flip(frame, 1)
        if success is not None:
            frame = detector.findHands(frame)
            listLmsAll = detector.findPosition(frame, maxNumHand=1, draw=False)
            if len(listLmsAll) > 0:
                point1 = (listLmsAll[0][4][1], listLmsAll[0][4][2])
                cv2.circle(frame, point1, 12, (255, 0, 255), -1, cv2.LINE_AA)
                point2 = (listLmsAll[0][8][1], listLmsAll[0][8][2])
                cv2.circle(frame, point2, 12, (255, 0, 255), -1, cv2.LINE_AA)

                x_middle = (point1[0] + point2[0]) // 2
                y_middle = (point1[1] + point2[1]) // 2
                cv2.circle(frame, (x_middle, y_middle), 15, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.line(frame, point1, point2, (255, 0, 255), 4, cv2.LINE_AA)

                distance = distance2Point(point1, point2)
                if distance < 50:
                    cv2.circle(frame, (x_middle, y_middle), 15, (0, 255, 0), -1, cv2.LINE_AA)
                elif distance > 320:
                    cv2.circle(frame, (x_middle, y_middle), 15, (255, 255, 255), -1, cv2.LINE_AA)
                vol = np.interp(distance, [50, 320], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

                cv2.putText(frame, "Volume", (45, 680), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (50, 250), (100, 650), (255, 255, 255), 2, cv2.LINE_AA)

                taskbar = int(np.interp(distance, [50, 320], [0, 400]))
                cv2.rectangle(frame, (50, (650 - taskbar)), (100, 650), (255, 255, 255), -1, cv2.LINE_AA)

                percent = int(np.interp(distance, [50, 320], [0, 100]))
                cv2.putText(frame, str(percent) + "%", (53, (630 - taskbar)), cv2.FONT_HERSHEY_PLAIN, 1.5,
                            (255, 255, 255), 2, cv2.LINE_AA)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(frame, "FPS:" + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow("Volume Control", frame)
            if cv2.waitKey(8) & 0xFF == 113:
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
