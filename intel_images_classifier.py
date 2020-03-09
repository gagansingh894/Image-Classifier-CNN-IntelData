import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import tqdm
import pickle

# Defining paths and creating empty lists to store images
#DATADIR = r'D:\Work\Kaggle\IntelImages\Data' #windows path
DATADIR = r'/media/gagandeep/2E92405C92402AA3/Work/Kaggle/IntelImages/Data'
CATEGORIES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
CATEGORYMAPPER = {'buildings':0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
IMG_SIZE = 100
X = []
y = []
datadict = {}

#lopping through the folders
for category in tqdm.tqdm(CATEGORIES):
	#Create a dictionary which stores number of examples for each class
	datadict[category] = int(len(os.listdir(os.path.join(DATADIR, category))))
	path = os.path.join(DATADIR, category)
	for img in tqdm.tqdm(os.listdir(path)):
		img_array = cv2.imread(os.path.join(path, img))
		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
		new_array = new_array/255.0
		X.append(new_array)
		y.append(CATEGORYMAPPER[category])

print(datadict)

y = pd.get_dummies(y)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)
print(len(X))
print(len(y))

# pickle_out = open(DATADIR + r"/X.pickle", "wb")
# pickle.dump(X, pickle_out, protocol=4)
# pickle_out.close()

# pickle_out = open(DATADIR + r"/y.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

#Spliting the data into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
	random_state=42, stratify=y)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

# Model Building
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
model.compile(loss='cross_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=5)
