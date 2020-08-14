from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import tensorflowjs as tfjs

DATADIR = os.getcwd()
CATEGORYMAPPER = {'buildings':0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
PATH = r'/media/gagandeep/2E92405C92402AA3/Work/Kaggle/IntelImages/Data/buildings/0.jpg'
keysList = list(CATEGORYMAPPER.keys())
IMG_SIZE = 100

model = load_model(DATADIR + '/model_3CL_32_64_128_3_2_3DL_Dropout_SoftMax.h5', )
img = cv2.imread(PATH)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img/255.0
pred_val = keysList[int(model.predict_classes(np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)))]
print(pred_val)
tfjs.converters.save_keras_model(model, DATADIR + '/model_3CL_32_64_128_3_2_3DL_Dropout_SoftMax')


