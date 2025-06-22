import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

st.set_page_config(page_title="Live Hand Sign Recognition", layout="wide")
st.title("🎥 Live Hand Sign Recognition (A, B, C)")

# Load model and labels
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["A", "B", "C"]

offset = 20
imgSize = 300

class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = detector
        self.classifier = classifier

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        hands, _ = self.detector.findHands(img, draw=False)
        if hands:
            x, y, w, h = hands[0]['bbox']
            # cropping with safety
            y1, y2 = max(0, y-offset), min(img.shape[0], y+h+offset)
            x1, x2 = max(0, x-offset), min(img.shape[1], x+w+offset)
            imgCrop = img[y1:y2, x1:x2]
            if imgCrop.size:
                # prepare white canvas
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                aspectRatio = (h) / (w)
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # classify
                _, index = self.classifier.getPrediction(imgWhite, draw=False)
                letter = labels[index]

                # draw UI
                cv2.rectangle(img, (x1, y1-50), (x1+90, y1), (255,0,255), cv2.FILLED)
                cv2.putText(img, letter, (x1, y1-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 4)

        return img

# Start webcam streamer
webrtc_streamer(
    key="hand-sign",
    video_transformer_factory=SignLanguageTransformer,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{
   "urls": [ "stun:bn-turn2.xirsys.com" ]
}, {
   "username": "rJhsnYer6656GcqAKdFv-Z3-h8aNNR6PkqNzxkF776vL4EUhx0bJaEjH4rQmjjLqAAAAAGhXjPFTaWduTGFuZ3VhZ2U=",
   "credential": "3b64b224-4f25-11f0-8c9c-0242ac140004",
   "urls": [
       "turn:bn-turn2.xirsys.com:80?transport=udp",
       "turn:bn-turn2.xirsys.com:3478?transport=udp",
       "turn:bn-turn2.xirsys.com:80?transport=tcp",
       "turn:bn-turn2.xirsys.com:3478?transport=tcp",
       "turns:bn-turn2.xirsys.com:443?transport=tcp",
       "turns:bn-turn2.xirsys.com:5349?transport=tcp"
   ]
}]},
)
