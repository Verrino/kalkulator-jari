from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
import pandas as pd

app = Flask(__name__)

# 0 = webcam, 1 = other camera output
cap = cv2.VideoCapture(0)

model = joblib.load('../trained_model_sklearn.pkl')

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

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return binary

def predict(result):
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

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    success, frame = cap.read()
    if success:
        result = preprocess(frame)
        y_pred = predict(result)[0]
        return jsonify(prediction=str(y_pred))
    return jsonify(prediction="Error")

@app.route('/arithmetic', methods=['POST'])
def arithmetic():
    data = request.get_json()
    operation = data.get('operation')
    num1 = float(data.get('num1'))
    num2 = float(data.get('num2'))
    result = 0
    if operation == 'add':
        result = num1 + num2
    elif operation == 'subtract':
        result = num1 - num2
    elif operation == 'multiply':
        result = num1 * num2
    elif operation == 'divide':
        if num2 != 0:
            result = num1 / num2
        else:
            return jsonify(result="Error: Division by zero")
    return jsonify(result=str(result))

if __name__ == '__main__':
    app.run(debug=True)
