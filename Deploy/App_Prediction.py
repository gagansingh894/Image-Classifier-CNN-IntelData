import tkinter
import tkinter.filedialog
from tkinter import *
from tkinter import messagebox as tkMessageBox
from PIL import ImageTk,Image
from keras.models import load_model
import cv2
import os
import numpy as np
import ctypes

DATADIR = os.getcwd()
CATEGORYMAPPER = {'buildings':0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
keysList = list(CATEGORYMAPPER.keys())
IMG_SIZE = 100
global path

model = load_model(DATADIR + '/model_3CL_32_64_128_3_2_3DL_Dropout_SoftMax.h5')

def browseFile():
	global path
	root = Tk()
	root.withdraw() #use to hide tkinter window
	a=tkinter.filedialog
	path= a.askopenfilename(filetypes=[("Image File",'.jpg')])
	load = Image.open(path)
	render = ImageTk.PhotoImage(load)
	imgfile = Label(image=render)
	imgfile.image = render
	imgfile.place(x=70, y=70)

def predict():	
	img = cv2.imread(path)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	img = img/255.0
	pred_val = keysList[int(model.predict_classes(np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)))]
	txt = "The picture is of {}".format(pred_val)
	tkMessageBox.showinfo("Result", txt)

my_window = Tk()
my_window.geometry('{}x{}'.format(300,270))
button_1 = Button(my_window, text='Browse', command=browseFile)
button_2 = Button(my_window, text='Predict', command=predict)
button_1.pack(side=TOP)
button_2.pack(side=TOP)
my_window.mainloop()