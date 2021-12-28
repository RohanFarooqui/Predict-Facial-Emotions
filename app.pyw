#############################################
### Author      : M.ROHAN FAROOQUI          #
##  Application : Predict Facial Emotions   #
##  File        : app.py                    #
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

###> Machine Learning Libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam  
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

###> Emotion Model Load
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
#> Load Weights from Modal File
emotion_model.load_weights('train_modal/epochs_1000_model.h5')

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
        bounding_box = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
        #> Emotion List
        emotionDictionary = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
        #> Emotion Image Links
        emotionImagePath   ={0:"emojis/angry.png",2:"emojis/disgusted.png",2:"emojis/fearful.png",3:"emojis/happy.png",4:"emojis/neutral.png",5:"emojis/sad.png",6:"emojis/surpriced.png"}

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maximumIndex = int(np.argmax(emotion_prediction))
             
        
        if flag is None:
            messagebox.showinfo("Error ⚠", "Error in capturaing camera feed .. !!")
        elif flag:
            self.last_frame = frame.copy()
            pic = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)     
            img = Image.fromarray(pic)
            
            imgtk = ImageTk.PhotoImage(image=img)

            #> Set camera feed on Label
            self.cameraFeed.imgtk = imgtk
            self.cameraFeed.configure(image=imgtk)
            
            try:
                #> Set Camera Text on Label
                self.cameraLabel.configure(text=emotionDictionary[maximumIndex],font=('arial',10,'bold'))
                #> Set Emoji on Label
                frame2=cv2.imread(emotionImagePath[maximumIndex])
                frame2 = cv2.resize(frame2,(300,300))
                pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
                img2=Image.fromarray(frame2)
                imgtk2=ImageTk.PhotoImage(image=img2)
                self.emoji.imgtk2=imgtk2
                self.emoji.configure(image=imgtk2)
                #> Set Emoji Text on Label
                self.emojiLabel.configure(text=emotionDictionary[maximumIndex],font=('arial',10,'bold')) 
            except:
                pass
            
            self.cameraFeed.after(1,self.captureOpenCvVideo)  
    
    ###> Exit Main Window       
    def exitWindow(self):
        if messagebox.askokcancel("Quit", "Do you want to quit ?"):
            self.root.destroy()            

    ###> Init GUI Window
    def init_gui(self):
        """Builds GUI."""
        ##> Initialiaze Window in Center of Screen
        window_width  = 750
        window_height = 500
        positionRight = int(self.root.winfo_screenwidth()/2 - window_width/2)
        positionDown = int(self.root.winfo_screenheight()/2 -  window_height/2)
        self.root.geometry("+{}+{}".format(positionRight, positionDown))  
        ##> Title of Window   
        self.root.title('Predict Facial Emotions')
        ##> Window Sizes
        self.root.geometry("750x500")
        ##> Make Resizable False
        self.root.resizable(width=False, height=False)
        ##> Set Icon on Top Bar
        try: self.root.iconbitmap("Images\icon.ico")
        except: pass
        
        style = ThemedStyle(self.root)
        style.set_theme("radiance")
        
        ##> First box : Heading
        self.root.first_box = tk.Frame(self.root,relief=FLAT,borderwidth=1)
        #> Heading
        heading = tk.Label(self.root.first_box, text="Predict Facial Emotions",font = ("Times",20))
        heading.grid(row=0,column=6)
        self.root.first_box.place(relx=0.32, rely=0.009)        
        
        ##> Second Box : Preview Camera Feed 
        self.root.second_box = tk.Frame(self.root,relief=GROOVE,borderwidth=2)
        #> Show Camera Feed
        self.cameraFeed = tk.Label(self.root.second_box,width="350",height="350")
        self.cameraFeed.grid(row=1, column=1)
        #> Show Camera Feed Label
        self.cameraLabel = tk.Label(self.root.second_box)
        self.cameraLabel.grid(row=2, column=1)
        #> End Second Box
        self.root.second_box.place(relx=0.009, rely=0.11)  
        
        ##> Third Box : Preview Emoji
        self.root.third_box = tk.Frame(self.root,relief=GROOVE,borderwidth=2)
        #> Show Emoji
        self.emoji = tk.Label(self.root.third_box,width="350",height="350")
        self.emoji.grid(row=1, column=1)
        #> Show Emoji Label
        self.emojiLabel = tk.Label(self.root.third_box)
        self.emojiLabel.grid(row=2, column=1)
        #> End Third Box
        self.root.third_box.place(relx=0.51, rely=0.11)  
        
        
        ##> Fourth Box : Button for Exiting Main Window
        self.root.forth_box=Frame(self.root,relief=FLAT,borderwidth=1)
        #> Create Button
        self.button_generate= ttk.Button(self.root.forth_box,text="Exit",command=self.exitWindow)
        self.button_generate.grid(row=1,column=0, padx=10, pady=10)
        #> End Fourth Box
        self.root.forth_box.place(relx=0.78, rely=0.84, y=30)
       
        
        ##> Start Camera Feed 
        self.captureOpenCvVideo()


if __name__ == '__main__':
    root = tk.Tk()
    Main_Window(root)
    root.mainloop()
