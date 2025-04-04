# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import numpy as np
from pygame import mixer
import time

# ============================  DATASET PATH  ============================ #
dataset_path = '/home/harsh/Downloads/DrowsyShield /yawn/dataset_new'
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

# ============================  IMAGE DATA LOADING  ============================ #
img_size = (24, 24)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=img_size, batch_size=batch_size, 
    class_mode='binary', color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    test_path, target_size=img_size, batch_size=batch_size, 
    class_mode='binary', color_mode='grayscale'
)

# ============================  CNN MODEL DEFINITION  ============================ #
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary Classification (Open/Closed)
])

# ============================  TRAINING THE MODEL  ============================ #
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=test_generator, epochs=5)

# ============================  SAVING THE MODEL  ============================ #
model_save_path = '/home/harsh/Downloads/DrowsyShield /yawn/models/cnnCat2.keras'
model.save(model_save_path)

# ============================  LOAD THE MODEL  ============================ #
model = load_model(model_save_path)

# ============================  HAAR CASCADE LOAD  ============================ #
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# ============================  LOAD ALARM SOUND  ============================ #
mixer.init()
sound = mixer.Sound('/home/harsh/Downloads/DrowsyShield /ALARME2.WAV')

# ============================  OPEN WEBCAM  ============================ #
cap = cv2.VideoCapture(0)
score = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Warning: Could not read frame from webcam. Exiting...")
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    rpred, lpred = [1], [1]  # Default to "Open" if no eyes are detected

    for (x, y, w, h) in right_eye:
        r_eye = gray[y:y+h, x:x+w]
        r_eye = cv2.resize(r_eye, (24, 24)) / 255.0
        r_eye = r_eye.reshape(1, 24, 24, 1)
        rpred = np.argmax(model.predict(r_eye))
        break

    for (x, y, w, h) in left_eye:
        l_eye = gray[y:y+h, x:x+w]
        l_eye = cv2.resize(l_eye, (24, 24)) / 255.0
        l_eye = l_eye.reshape(1, 24, 24, 1)
        lpred = np.argmax(model.predict(l_eye))
        break

    # ============================  DROWSINESS DETECTION LOGIC  ============================ #
    if rpred == 0 and lpred == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        try:
            sound.play()
        except:
            pass
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 3)

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

