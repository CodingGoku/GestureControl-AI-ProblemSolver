import cv2
from HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
from gtts import gTTS
import pygame
import time


# Initialize pygame for audio
pygame.mixer.init()

instructions = (
        "Welcome to the gesture-based application. "
        "To start, Point your index finger up to draw on the canvas."
        "Hold up your thumb to clear the screen."
        "Hold first three fingers and the thumb to trigger the AI."
    )

# Function to play instructions
def play_instructions():
    tts = gTTS(text=instructions, lang='en')
    tts.save("instructions.mp3")
    pygame.mixer.music.load("instructions.mp3")

    # Set playback speed - higher value for faster speed
    pygame.mixer.music.set_endevent(pygame.USEREVENT)  # Custom event to detect playback end
    pygame.mixer.music.play()

    # Use this to increase playback speed by increasing frequency
    pygame.mixer.music.set_volume(1.5)  # Example: 2x speed

    # Wait until playback is complete
    while pygame.mixer.music.get_busy():
        pygame.time.delay(100)
    
    pygame.mixer.music.stop()


st.title(instructions)
# Play instructions at the start
play_instructions()


# Streamlit setup
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Initialize Google GenAI
genai.configure(api_key="AIzaSyBGs08t8EM5HHhNzCBfLkl9PaxPz6TiBGA")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)  # primary camera
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Function to get hand information
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

# Draw function for canvas
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # index finger up
        current_pos = lmList[8][0:2]  # index finger tip position
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (102, 102, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:  # thumb up
        canvas = np.zeros_like(img)  # clear canvas
    return current_pos, canvas


def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

    # Wait until playback is complete
    while pygame.mixer.music.get_busy():
        continue

    # Remove the audio file and clean up mixer
    # os.remove("output.mp3")
    pygame.mixer.quit()

# Send image and text to AI and get response with TTS
def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # trigger AI when first 4 fingers are up
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        output_text = response.text
        text_to_speech(output_text)  # Read answer aloud
        return output_text
    return ""

# Main loop
prev_pos = None
canvas = None
image_combined = None
output_text = ""

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

        if output_text:
            output_text_area.text(output_text)

    image_combined = cv2.addWeighted(img, 0.6, canvas, 0.4, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    cv2.waitKey(1)
