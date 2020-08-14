from posix import listdir
from flask import Flask, jsonify, request, redirect, url_for, flash
from flask.templating import render_template
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import glob
import numpy as np
import cv2
import shutil

DIRPATH = os.path.dirname(os.path.realpath(__file__))
CATEGORYMAPPER = {'buildings':0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
IMG_SIZE = 100
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = DIRPATH + '/static'

app = Flask(__name__, root_path=DIRPATH)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

model = load_model(DIRPATH + r'/model_3CL_32_64_128_3_2_3DL_Dropout_SoftMax.h5')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', value='', file='')

    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST','GET'])
def upload_image_and_predict():
    global file, img, filename
    if len(os.listdir(UPLOAD_FOLDER)) != 0:
        shutil.rmtree(UPLOAD_FOLDER)

    if request.method == 'POST':
        print('Start')
        # check if the post request has the file part
        if 'myImage' not in request.files:
            print('here')
            flash('No file part')
            return redirect(request.url)
        file = request.files['myImage']
        # if user does not select file, browser also
        # submit an empty part without filename
        print('here')
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print('here')
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file = max(glob.glob(UPLOAD_FOLDER + r'/*'), key=os.path.getctime)
            print(file)
            img = cv2.imread(file)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img/255.0
    
    keysList = list(CATEGORYMAPPER.keys())
    pred = keysList[int(model.predict_classes(np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)))]
    return render_template('index.html', value="The picture is of {}".format(pred), filename=filename)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)