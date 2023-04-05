import cv2
import time
import cvzone
import random
import HandTrackingModule as htm

timer = 0
stateResult = False
startGame = False
scores = [0, 0]
imgAI = None
pathImgBG = "image/BG.png"
pathimgAI = "image/{}.png"

cap = cv2.VideoCapture(0)
detector = htm.HandDetector(maxNumHands=1)

while True:
    imgBG = cv2.imread(pathImgBG)
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (0, 0), None, 0.875, 0.875)
    frame = frame[:, 80:480]

    frame = detector.findHands(frame)
    listLmsAll = detector.findPosition(frame)
    if startGame:
        if not stateResult:
            timer = time.time() - cTime
            cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

        if timer > 3:
            startGame = False
            stateResult = True
            timer = 0
            checkResult = True

        if checkResult:
            checkResult = False
            player = None
            if len(listLmsAll) > 0 :
                fingers,_ = detector.fingersUp(listLmsAll[0])
                if fingers == [0, 0, 0, 0, 0]:
                    player = 1
                if fingers == [1, 1, 1, 1, 1]:
                    player = 2
                if fingers == [0, 1, 1, 0, 0]:
                    player = 3

                AI = random.randint(1, 3)
                imgAI = cv2.imread(pathimgAI.format(AI), cv2.IMREAD_UNCHANGED)
                imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

                # Player Wins
                if (player == 1 and AI == 3) or (player == 2 and AI == 1) or (player == 3 and AI == 2):
                    scores[1] += 1

                # AI Wins
                if (player == 3 and AI == 1) or (player == 1 and AI == 2) or (player == 2 and AI == 3):
                    scores[0] += 1

    imgBG[234:654, 795:1195] = frame

    if stateResult and imgAI is not None:
        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    cv2.imshow("Rock_Paper_Scissors", imgBG)

    key = cv2.waitKey(8)
    if key == ord('s'):
        startGame = True
        cTime = time.time()
        stateResult = False
        checkResult = False
    if key & 0xFF == 113:
        break

cap.release()
cv2.destroyAllWindows()
