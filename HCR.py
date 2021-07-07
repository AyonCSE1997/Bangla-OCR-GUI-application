import tensorflow as tf
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("model_hand_reco.h5")

from tkinter import *
from PIL import ImageGrab
from PIL import ImageTk, Image
import imageio
import tkinter.font as font
from tkinter import filedialog
import os


class Paint(object):
    def __init__(self):
        self.root=Tk()
        self.root.title('HCR')
        #self.root.wm_iconbitmap('44143.ico')
        self.root.configure(background='light salmon')
        #self.c=Canvas(self.root,bg='white',height=330,width=400)
        #self.label=Label(self.root,text='Draw any numer',font=20,bg='light salmon')
        #self.label.grid(row=0,column=3)
        #self.c.grid(row=1,columnspan=9)
#         self.c.create_line(0,0,400,0,width=20,fill='midnight blue')
#         self.c.create_line(0,0,0,330,width=20,fill='midnight blue')
#         self.c.create_line(400,0,400,330,width=20,fill='midnight blue')
#         self.c.create_line(0,330,400,330,width=20,fill='midnight blue')
        self.myfont=font.Font(size=20,weight='bold')
        self.predicting_button=Button(self.root,text='Select & Predict',fg='maroon',bg='steel blue',height=2,width=16,font=self.myfont,command=lambda:self.classify(self.root))
        self.predicting_button.grid(row=2,column=3)
        #self.clear=Button(self.root,text='Clear',fg='blue',bg='red',height=2,width=6,font=self.myfont,command=self.clear)
        #self.clear.grid(row=2,column=5)
        self.prediction_text = Text(self.root, height=5, width=5)
        self.prediction_text.grid(row=4, column=3)
        self.label=Label(self.root,text="Predicted Number is",fg="black",font=30,bg='light salmon')
        
        self.label.grid(row=3,column=3)
        self.model=model
        self.setup()
        self.root.mainloop()
    def setup(self):
        self.old_x=None
        self.old_y=None
        self.color='black'
        self.linewidth=8
        #self.c.bind('<B1-Motion>', self.paint)
        #self.c.bind('<ButtonRelease-1>', self.reset)
#     def paint(self,event):
#         paint_color=self.color
#         if self.old_x and self.old_y:
#             self.c.create_line(self.old_x,self.old_y,event.x,event.y,fill=paint_color,width=self.linewidth,capstyle=ROUND,
#                               smooth=TRUE,splinesteps=48)
#         self.old_x=event.x
#         self.old_y=event.y
    def clear(self):
        """Clear drawing area"""
        #self.c.delete("all")

    def reset(self, event):
        """reset old_x and old_y if the left mouse button is released"""
        self.old_x, self.old_y = None, None    



    def openfn(self):
        filename = filedialog.askopenfilename(title='open')
        return filename

    def classify(self,widget):
       
        path = self.openfn()
        img = cv2.imread(path)
        img_copy = img.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400,440))
        #plt.imshow(img, cmap='binary')
        #plt.show()
        

        img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
        img_final = cv2.resize(img_thresh, (28,28))
        #plt.imshow(img_final, cmap='binary')
        #plt.show()
        img_final =np.reshape(img_final, (1,28,28,1))

        pred = self.model.predict([img_final])

        # Get index with highest probability
        pred = np.argmax(pred)
        #print(pred)
        self.prediction_text.delete("1.0", END)
        self.prediction_text.insert(END, pred)

        labelfont = ('times', 30, 'bold')
        self.prediction_text.config(font=labelfont)
        
if __name__ == '__main__':
    Paint()