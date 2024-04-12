from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
from deepface import DeepFace

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = r'C:\Users\91832\Downloads\gender_detection.model'
model1=load_model(r"D:\projects\age_gender\modelgender.h5")

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/home')

classes = ['man','woman']

@app.route('/home', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route("/knowus")
def know():
    return render_template('knowus.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("hi")
    if request.method == 'POST':
        print("SEKHAR")
        # Get the file from post request
        f = request.files['fileq']

        # Save the file to ./uploads
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(224, 224))

        # Preprocessing the image
        x = image.img_to_array(img)
        # x = np.true_divide(x, 255)

        # Be careful how your trained model deals with the input
        # otherwise, it won't make correct prediction!

        label2=DeepFace.analyze(x,actions=['age'],enforce_detection=False)['age']
        label1=DeepFace.analyze(x,actions=['gender'],enforce_detection=False)['gender']
        if(label2 in range(0,11)):
            label2=" around 1-10 age"
        elif(label2 in range(11,21)):
            label2="around 10-20 age"
        elif(label2 in range(21,31)):
            label2="around 20-30 age"
        elif(label2 in range(31,41)):
            label2="around 30-40 age"
        elif(label2 in range(41,51)):
            label2="around 40-50 age"
        elif(label2 in range(51,61)):
            label2="around 50-60 age"
        elif(label2 in range(61,71)):
            label2="around 60-70 age"
        elif(label2 in range(71,81)):
            label2="around 70-80 age"
        elif(label2 in range(81,91)):
            label2="around 80-90 age"
        elif(label2 in range(91,110)):
            label2="around 90-100 age"

        face_crop = cv2.resize(x, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        l=[]
        l.append(label1)
        # apply gender detection on face
        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]
        l.append(label)
        conf1 = model.predict(face_crop)[0]
        idx1 = np.argmax(conf1)
        l.append(classes[idx1])
        a=l.count("Man")
        b=l.count("Woman")
        if(a>b):
            ans="M"
        else:
            ans="F"

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        return render_template('indexr.html',data='AGE Around '+str(label2)+" GENDER "+str(ans))
    return None


if __name__ == '__main__':
    app.run(debug=True)
    