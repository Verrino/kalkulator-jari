import cv2
import time
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
import pandas as pd

# 0 = webcam, 1 = other camera output
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

capture_duration = 3

start_time = time.time()

cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)

# def preprocess(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     resized = cv2.resize(hsv, (128, 128))

#     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)

#     mask = cv2.inRange(resized, lower_skin, upper_skin)

#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)

#     _, threshold = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     return threshold

def preprocess(image):  
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    resized = cv2.resize(ycrcb, (128, 128))

    Cb_min, Cb_max = 77, 120
    Cr_min, Cr_max = 137, 163
    mask = cv2.inRange(resized, (0, Cr_min, Cb_min), (255, Cr_max, Cb_max))

    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def predict(result):
    model = joblib.load('trained_model_sklearn.pkl')

    contrasts = []
    homogeneities = []
    energies = []
    correlations = []
    dissimilarities = []

    distances = [1]
    angles = [0, np.pi/4, np.pi/2, np.pi*3/4]

    glcm = graycomatrix(image=result, distances=distances, angles=angles, levels=256)

    contrasts.append(graycoprops(glcm, 'contrast').mean())
    homogeneities.append(graycoprops(glcm, 'homogeneity').mean())
    energies.append(graycoprops(glcm, 'energy').mean())
    correlations.append(graycoprops(glcm, 'correlation').mean())
    dissimilarities.append(graycoprops(glcm, 'dissimilarity').mean())

    df = pd.DataFrame({
        'contrasts': contrasts,
        'homogeneities': homogeneities,
        'energies': energies,
        'correlations': correlations,
        'dissimilarities': dissimilarities
    })

    y_pred = model.predict(df)

    return y_pred

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow('Video Feed', frame)

    if time.time() - start_time >= capture_duration:
        result = preprocess(frame)

        cv2.imshow("Frame", result)
        y_pred = predict(result)[0]

        print(y_pred)
        
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
