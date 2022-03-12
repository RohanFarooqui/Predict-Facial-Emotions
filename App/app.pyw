#############################################
### Author      : M.ROHAN FAROOQUI          #
##  Application : Predict Facial Emotions   #
##  File        : app.pyw                   #
############################################# 

#### Imports

###> Tkinter Libraries
from tkinter import *
import tkinter.ttk as ttk
import tkinter as tk
#MSG dialog box
from tkinter import messagebox
#Ask to save File
from tkinter.filedialog import asksaveasfile
from tkinter  import simpledialog
#Tkinter Theme File 
from ttkthemes import ThemedStyle

#For handling Images
from PIL import Image , ImageTk

###> For Showing Feed 
import cv2
import numpy as np

###> Other Libs
import os

###> Machine Learning Libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam  
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

###> Emotion Model Load
#> Disable Tensor Flow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#> Sequential : allows you to create models
emotionsModel = Sequential()
#> Conv2D : Creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
#> Kernal Size : An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
#> activation relu : Use to neglact values less than or equal to ZERO 
emotionsModel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotionsModel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotionsModel.add(MaxPooling2D(pool_size=(2, 2)))
#> Dropout : To Prevent Neural Networks from Overfitting
emotionsModel.add(Dropout(0.25))
emotionsModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#> MaxPooling2D : Take the maximum value over an input window size
emotionsModel.add(MaxPooling2D(pool_size=(2, 2)))
emotionsModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotionsModel.add(MaxPooling2D(pool_size=(2, 2)))
emotionsModel.add(Dropout(0.25))
#> Flatten take array of elements and convert into 1D
emotionsModel.add(Flatten())
#> Dense : It feeds all outputs from the previous layer to all its neurons, each neuron providing one output to the next layer.
emotionsModel.add(Dense(1024, activation='relu'))
emotionsModel.add(Dropout(0.5))
emotionsModel.add(Dense(7, activation='softmax'))
#> Load Weights from Modal File
emotionsModel.load_weights('Data/epochs_5000_model.h5')
#> Capture Feed from Video Cam
cap    = cv2.VideoCapture(0)


### Code Start HERE ###
class Main_Window(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        #> Init GUI
        self.init_gui()
        #> Capture Video Initially
        cv2.ocl.setUseOpenCL(False)
        self.last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
         
    ###> Capture Feed and Predict   
    def captureOpenCvVideo(self):
        #> Read Camera Feed
        global cap
        #> CV - Capture Feed from Camera
        flag, frame = cap.read()
        #> CV - Resize Frame
        frame = cv2.resize(frame,(300,300))
        #> Cascade Classifier : used to detect objects in other images .. In this case use to detect the bounding boxes of face in the webcam.
        bounding_box = cv2.CascadeClassifier('Data/haarcascade/haarcascade_frontalface_default.xml')
        #> Emotion List
        emotionDictionary = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
        #> Emotion Image Links
        emotionImagePath  = {0:"Data/emojis/angry.png",2:"Data/emojis/disgusted.png",2:"Data/emojis/fearful.png",3:"Data/emojis/happy.png",4:"Data/emojis/neutral.png",5:"Data/emojis/sad.png",6:"emojis/surpriced.png"}
        #> cvtColor()     : convert an image from one color space to another
        #> COLOR_BGR2GRAY : convert our original image from the BGR color space to gray
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #> Bounding_box.detectMultiScale : Detects objects of different sizes in the input image in this case it is face.
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        # Draw a rectangle around the faces
        for (x, y, w, h) in num_faces:
            #> Create Rectangle around face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            #> Predict emotions using emotion model
            emotion_prediction = emotionsModel.predict(cropped_img)
            #> Give Index for emotion
            maximumIndex = int(np.argmax(emotion_prediction))
             
        #> If there is any error in CAM or CAM not available it will show Error Message Box
        if flag is None:
            messagebox.showinfo("Error ⚠", "Error in capturaing camera feed .. !!")
        elif flag:
            #> Copy Frame
            self.last_frame = frame.copy()
            #> Convert Frame from the BGR color space to gray
            pic = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB) 
            #> Create Image Array    
            img = Image.fromarray(pic)
            #> Show Image on Tkinter PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)

            #> Set camera feed on Label
            self.cameraFeed.imgtk = imgtk
            self.cameraFeed.configure(image=imgtk)
            
            try:
                #> Set Camera Text on Label
                self.cameraLabel.configure(text=emotionDictionary[maximumIndex],font=('arial',10,'bold'))
                #> Set Emoji on Label
                frame2=cv2.imread(emotionImagePath[maximumIndex])
                #> Resize Frame
                frame2 = cv2.resize(frame2,(300,300))
                #> Convert Frame from the BGR color space to gray
                pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
                #> Create Image Array
                img2=Image.fromarray(frame2)
                #> Show Image on Tkinter PhotoImage
                imgtk2=ImageTk.PhotoImage(image=img2)
                self.emoji.imgtk2=imgtk2
                self.emoji.configure(image=imgtk2)
                #> Set Emoji Text on Label
                self.emojiLabel.configure(text=emotionDictionary[maximumIndex],font=('arial',10,'bold')) 
            except:
                pass
            
            # Delay, in milliseconds : Update Camera Feed After 4  milliseconds 
            self.cameraFeed.after(4,self.captureOpenCvVideo)  
    
    ###> Exit Main Window       
    def exitWindow(self):
        #> Ask to Exit or not , if yes then it will destroy main window
        if messagebox.askokcancel("Quit", "Do you want to quit ?"):
            cap.release()
            self.root.destroy()            

    ###> Init GUI Window
    def init_gui(self):
        """Builds GUI."""
        ##> Initialiaze Window in Center of Screen
        #> Window Width
        window_width  = 750
        #> Window Height
        window_height = 500
        #> Set Window in Center from Right
        positionRight = int(self.root.winfo_screenwidth()/2 - window_width/2)
        #> Set Window in Center from Down
        positionDown = int(self.root.winfo_screenheight()/2 -  window_height/2)
        #> Center Window
        self.root.geometry("+{}+{}".format(positionRight, positionDown))  
        ##> Title of Window   
        self.root.title('Predict Facial Emotions')
        ##> Window Sizes
        self.root.geometry("750x500")
        ##> Make Resizable False
        self.root.resizable(width=False, height=False)
        ##> Set Icon on Top Bar
        try: self.root.iconbitmap("Data/Logo/logo.ico")
        except: pass
        
        style = ThemedStyle(self.root)
        style.set_theme("radiance")
        
        ##> First box : Heading
        self.root.first_box = tk.Frame(self.root,relief=FLAT,borderwidth=1)
        #> Set Logo
        self.root.logo_1 = PhotoImage(file = 'Data/Logo/logo.png')
        label = Label(self.root.first_box,image=self.root.logo_1,width="50",height="50")
        label.grid(row=0, column=1)
        label.image = self.root.logo_1
        #> Heading
        heading = tk.Label(self.root.first_box, text="Predict Facial Emotions",font = ("Times",20))
        heading.grid(row=0,column=2)
        ##> Place First Box  on GUI 
        self.root.first_box.place(relx=0.32, rely=0.009)        
        
        ##> Second Box : Preview Camera Feed 
        self.root.second_box = tk.Frame(self.root,relief=GROOVE,borderwidth=2)
        #> Show Camera Feed
        self.cameraFeed = tk.Label(self.root.second_box,width="350",height="350")
        self.cameraFeed.grid(row=1, column=1)
        #> Show Camera Feed Label
        self.cameraLabel = tk.Label(self.root.second_box)
        self.cameraLabel.grid(row=2, column=1)
        ##> Place Second Box  on GUI 
        self.root.second_box.place(relx=0.009, rely=0.15)  
        
        ##> Third Box : Preview Emoji
        self.root.third_box = tk.Frame(self.root,relief=GROOVE,borderwidth=2)
        #> Show Emoji
        self.emoji = tk.Label(self.root.third_box,width="350",height="350")
        self.emoji.grid(row=1, column=1)
        #> Show Emoji Label
        self.emojiLabel = tk.Label(self.root.third_box)
        self.emojiLabel.grid(row=2, column=1)
        ##> Place Third Box  on GUI 
        self.root.third_box.place(relx=0.51, rely=0.15)  
        
        
        ##> Fourth Box : Button for Exiting Main Window
        self.root.forth_box=Frame(self.root,relief=FLAT,borderwidth=1)
        #> Create Button
        self.button_generate= ttk.Button(self.root.forth_box,text="Exit",command=self.exitWindow)
        self.button_generate.grid(row=1,column=0)
        ##> Place Fourth Box  on GUI 
        self.root.forth_box.place(relx=0.82, rely=0.853, y=30)
       
        
        ##> Start Camera Feed 
        self.captureOpenCvVideo()


if __name__ == '__main__':
    root = tk.Tk()
    Main_Window(root)
    root.mainloop()
