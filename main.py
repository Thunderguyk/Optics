import cv2
import mediapipe as mp
import numpy as np
from transformers import pipeline
import pyttsx3
import time

# ========== Setup ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
grammar_corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Speech speed
engine.setProperty('volume', 1.0)  # Volume level

# ========== Dummy Classifier ==========
def classify_gesture(landmark_list):
    # Simulate predictions for testing
    words = ["hello", "my", "name", "is", "karthik", "and", "i", "love", "physics", "bro"]
    return np.random.choice(words)

# ========== Sign Language Conversion ==========
def run_sign_language_converter(max_words=7, frames_per_word=25):
    cap = cv2.VideoCapture(0)
    collected_words = []
    frame_count = 0

    print("🎬 Running Sign to English. Press Q to exit.")

    while len(collected_words) < max_words:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                frame_count += 1
                if frame_count % frames_per_word == 0:
                    word = classify_gesture(landmarks)
                    collected_words.append(word)
                    print(f"[+] Detected: {word}")

        cv2.imshow("Sign Language Converter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ========== Convert to Full Sentence ==========
    raw_sentence = ' '.join(collected_words)
    print(f"\n📝 RAW: {raw_sentence}")

    corrected = grammar_corrector(raw_sentence)[0]['generated_text']
    print(f"✅ FIXED: {corrected}")

    # ========== Speak It ==========
    print("\n🗣️ Speaking...")
    engine.say(corrected)
    engine.runAndWait()
    print("✅ Done.")

# ========== Run ==========
run_sign_language_converter()
