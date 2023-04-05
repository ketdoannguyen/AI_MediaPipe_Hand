import cv2
import mediapipe as mp


class HandDetector():
    def __init__(self, model=False, maxNumHands=2, modelComplexity=1, minDetection=1, minTracking=1):
        self.model = model
        self.maxNumHands = maxNumHands
        self.modelComplexity = modelComplexity
        self.minDetection = minDetection
        self.minTracking = minTracking

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.model, self.maxNumHands, self.modelComplexity, self.minDetection,
                                        self.minTracking)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawType = mp.solutions.drawing_styles

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                           self.mpDrawType.get_default_hand_landmarks_style(),
                                           self.mpDrawType.get_default_hand_connections_style())
        return img

    def findPosition(self, img, maxNumHand=5, draw=False):
        maxNumHand = min(self.maxNumHands, maxNumHand)
        height, weight = img.shape[:2]
        if self.results.multi_hand_landmarks:
            handedness = self.results.multi_handedness
            if self.maxNumHands == 1:
                listLms = []
                handType = handedness[0].classification[0].label
                handLms = self.results.multi_hand_landmarks[0]
                for idx, lms in enumerate(handLms.landmark):
                    cx, cy = int(lms.x * weight), int(lms.y * height)
                    listLms.append([idx, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 12, (255, 0, 255), -1, cv2.LINE_AA)
                return list((listLms, handType))
            else:
                listLmsAllHand = []
                for idxHand, handLms in enumerate(self.results.multi_hand_landmarks[:maxNumHand]):
                    handType = handedness[idxHand].classification[0].label
                    listLmsEachHand = []
                    for idx, lms in enumerate(handLms.landmark):
                        cx, cy = int(lms.x * weight), int(lms.y * height)
                        listLmsEachHand.append([idx, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 8, (255, 0, 255), -1, cv2.LINE_AA)
                    listLmsAllHand.append((idxHand, listLmsEachHand, handType))
                return listLmsAllHand
        return []

    def fingersUp(self, listLms, handType="Right"):
        idxTips = [4, 8, 12, 16, 20]
        fingerList = []  # 1,0,0,0,0
        idxTipsTrue = []
        if handType == "Right":
            if listLms[4][1] <= listLms[3][1]:
                fingerList.append(1)
                idxTipsTrue.append(4)
            else:
                fingerList.append(0)
        else:  # Left
            if listLms[4][1] > listLms[3][1]:
                fingerList.append(1)
                idxTipsTrue.append(4)
            else:
                fingerList.append(0)
        for fingertip in idxTips[1:]:
            if listLms[fingertip][2] < listLms[fingertip - 2][2]:
                fingerList.append(1)
                idxTipsTrue.append(fingertip)
            else:
                fingerList.append(0)
        return fingerList, idxTipsTrue
