#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter #GUI Module
import cv2 # Computer Vision Module based on OpenCV
import PIL.Image,PIL.ImageTk # Python Image processing Module
import time # In-built module to get system time
import numpy as np # Numpy Module for statistiscal operations, face-recognition module dependency
import h5py as h5 # File format to store face encodings

from tkinter import filedialog, messagebox # GUI Module

import face_recognition # Python Module for Face Recognition Module based on dlib


# In[ ]:


#Create Storage file
f = h5.File('Data/Faces.h5','a')


# In[32]:


# Main Class for creating the application
class App():
    def __init__(self,window,window_title):# Main Window
        self.mainwindow = window
        self.mainwindow.title(window_title)# Window Title

        self.vid = MyVideoCapture(0) # Video Capture Class
        self.encodings = Encodings() # Face Recognition Class

        self.Page_One() # Main Page where Face recognition occurs

        self.mainwindow.resizable(0,0) # Disable Resizing of main window
        self.mainwindow.mainloop()

    def refresh(self,page):
        ret, self.image_frame = self.vid.get_frame() # Function to capture frame from camera along with location of face  
        self.face_frame,self.face_locations = self.vid.detect_face(self.image_frame)

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.cvtColor(self.face_frame, cv2.COLOR_BGR2RGB))) # Show the frame captured in the application
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        if page == 1:
            self.g = 1
        if page == 2:
            self.g = 2

    def update(self): # Update the main page
        
        self.refresh(1)

        self.frame1.after(self.delay, self.update)

    def update1(self): # Update the second page (Page where the faces are added to the database)
        
        self.refresh(2)

        if len(self.entry11.get()) == 0 or len(self.face_locations) == 0: # If there is no text in the textbox or no face is detected
            self.button12.config(state=tkinter.DISABLED)                  # Disable the button
        else:
            self.button12.config(state=tkinter.NORMAL) # Else enable it

        self.frame2.after(self.delay, self.update1)

    def text_clear(self): # Clear the textbox
        self.label1.config(text='')

        self.label1.after(3500,self.text_clear) # Wait for  seconds before clearing

    def Page_One(self): # Main Page
        mainwindow = self.mainwindow 

        self.frame1 = tkinter.Frame(master=mainwindow)
        self.frame1.pack()

        window = self.frame1

        self.label_1 = tkinter.Label(master=window,text="Face Recognize",justify=tkinter.CENTER)
        self.label_1.pack()

        self.canvas = tkinter.Canvas(master=window,width=self.vid.width,height=self.vid.height)
        self.canvas.pack()

        self.label1 = tkinter.Label(master=window,justify=tkinter.CENTER)
        self.label1.pack(side=tkinter.LEFT,expand=1,fill=tkinter.X,padx=5)

        self.button1 = tkinter.Button(master=window,text="Add",width=10,command=self.face_option)
        self.button1.pack(side=tkinter.RIGHT)

        self.delay = 15
        self.update()
        self.Get_Face()
        self.text_clear()

    def Page_Two(self): # Second Page
        mainwindow = self.mainwindow

        self.frame2 = tkinter.Frame(master=mainwindow)
        self.frame2.pack()

        window = self.frame2

        self.label_2 = tkinter.Label(master=window,text="Add Face",justify=tkinter.CENTER)
        self.label_2.pack()

        self.canvas = tkinter.Canvas(master=window,width=self.vid.width,height=self.vid.height)
        self.canvas.pack()

        self.label11 = tkinter.Label(master=window,text="Name",width=10,justify=tkinter.CENTER)
        self.label11.pack(side=tkinter.LEFT)

        self.entry11 = tkinter.Entry(master=window,justify=tkinter.LEFT)
        self.entry11.pack(side=tkinter.LEFT,expand=1,fill=tkinter.X,padx=5)

        self.button11 = tkinter.Button(master=window,text="Back",width=10,command=self.To_Page_One)
        self.button11.pack(side=tkinter.RIGHT)

        self.button12 = tkinter.Button(master=window,text="Add Face",width=10,command=self.Add_Face)
        self.button12.pack(side=tkinter.RIGHT)

        self.delay = 15
        self.update1()

    def To_Page_Two(self): # Function to goto page 1 (Main Page)
        self.frame1.destroy()
        self.face_select_window.destroy()
        self.Page_Two()

    def To_Page_One(self): # Function to goto page 2 (Second Page)
        self.frame2.destroy()
        self.Page_One()

    def face_option(self): # Function to create selection box, whether the face must be taken from camera or image
        self.face_select_window = tkinter.Toplevel(self.mainwindow)
        face_select_window = self.face_select_window
        face_select_window.geometry('300x100+200+200')
        face_select_window.title('Capture Face')
        
        self.button21 = tkinter.Button(master = face_select_window, text = "From Image", width = 10,command=self.browse_image)
        self.button21.pack(fill = tkinter.X, padx = 10, pady = 10)

        self.button22 = tkinter.Button(master = face_select_window, text = "From Camera", width = 10,command=self.To_Page_Two)
        self.button22.pack(fill = tkinter.X, padx = 10, pady = 10)

    def browse_image(self): # Image File Selector
        self.image_filename = filedialog.askopenfilename(parent = self.face_select_window, title = "Select Image File", initialdir = "/", 
                                           filetypes = [("Image Files",".jpg .jpeg .png .bmp .tiff"), 
                                             ("All files", " .*")])
        if len(self.image_filename) != 0:
            self.face_select_window.destroy()

        self.Get_Face_Image()

    def Get_Face(self): # Function to Detect face and recognize in Main Page
        if len(self.face_locations) != 0:
            self.candidate_encodings = self.encodings.get_encodings(self.image_frame,self.face_locations)
            if self.g == 1:
                message = self.encodings.match(self.candidate_encodings)
                self.label1.config(text=message)
                    
        self.frame1.after(5000,self.Get_Face)

    def Add_Face(self): # Function to Add face to the database from camera
        name = str(self.entry11.get())
        
        if len(self.face_locations) != 0:
            self.candidate_encodings = self.encodings.get_encodings(self.image_frame,self.face_locations)
            status = self.encodings.verify(name,self.candidate_encodings)
                
            if status:
                self.encodings.save_encodings(name,self.candidate_encodings )
                messagebox.showinfo(message="Person added to database")
            else:
                messagebox.showinfo(message="Person already exists in database")
        
    def Get_Face_Image(self): # Window to add face from image 
        if len(self.image_filename) != 0:
            self.face_add_window = tkinter.Toplevel(self.mainwindow)
            face_add_window = self.face_add_window
            face_add_window.geometry('300x75+200+200')
            face_add_window.title('Add Face from Image')

            image_file = self.image_filename # Get the file

            frame1 = tkinter.Frame(face_add_window)
            frame1.pack()

            frame2 = tkinter.Frame(face_add_window)
            frame2.pack(side = tkinter.BOTTOM)

            self.label31 = tkinter.Label(master=frame1,text=image_file)
            self.label31.pack()

            self.label32 = tkinter.Label(master=frame1,text="Name",width=10,justify=tkinter.CENTER)
            self.label32.pack(side=tkinter.LEFT)

            self.entry31 = tkinter.Entry(master=frame1,justify=tkinter.LEFT) # Textbox to give name
            self.entry31.pack(side=tkinter.LEFT,expand=1,fill=tkinter.X,padx=5)

            self.button31 = tkinter.Button(master=frame2,text="Add Face",width=10,command=self.Add_Face_Image_with_Name)
            self.button31.pack()

    def Add_Face_Image_with_Name(self): # Function to Add face to the database from image file
        name = str(self.entry31.get()) # Get the name

        self.face_add_window.destroy()

        image = face_recognition.load_image_file(self.image_filename) # Get the image file
        face_location = face_recognition.face_locations(image) # Locate the face

        if len(face_location) != 0:
            self.candidate_encodings = self.encodings.get_encodings(image,face_location) # Get the encodings
            status=self.encodings.verify(name,self.candidate_encodings)

            if status: # If no matched faces exist, add to database otherwise show that the face already exists
                self.encodings.save_encodings(name,self.candidate_encodings)
                messagebox.showinfo(message="Person added to database")
            else:
                messagebox.showinfo(message="Person already exists in database")

    def __del__(self): # Close the application
        f.close()

class MyVideoCapture():
    def __init__(self,source):
        self.vid = cv2.VideoCapture(source) # Get the feed from the camera, 0 indicates inbuilt camera

        if not self.vid.isOpened():
            raise ValueError("Unable to open Camera") # If problem with camera, raise error

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) # Dimensions
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret,frame = self.vid.read() # Read the feed
            return ret,frame

    def detect_face(self,frame): # Function to detect the face
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_locations =  face_locations[:1] 
        
        (top, right, bottom, left) = (0,0,0,0)

        for (top, right, bottom, left) in face_locations:# Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        return frame,face_locations

    def __del__(self): # Close the camera
        if self.vid.isOpened():
            self.vid.release()

class Encodings(): # Class to get the face encodings
    #Program to get embeddings from extracted face
    def get_encodings(self,frame,face_locations): # Function to get face encodings
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]
        return face_encodings

    def match(self,encoding): # Function to match the encodings with those stored in the databse
        if len(f.keys()) == 0:
            return "No Faces to Match"
        
        for k in list(f.keys()):
            known_embedding = f[k][:]
            status = face_recognition.compare_faces([known_embedding], encoding) # Compare the faces and determine whether they match
            status = status[0]
            if status: # If the faces match, send the name        
                return k
            else: # If faces don't match
                return "No Matches Found"
            
    def verify(self,name,encoding):  
        for k in list(f.keys()):
            if k == name:
                status = True
            else:
                known_embedding = f[k][:]
                status = face_recognition.compare_faces([known_embedding], encoding) # Compare the faces and determine whether they match
                status = status[0]
                
            if status:
                return False
        
        return True
            
    def save_encodings(self,name,encoding): # Save the encodings to the databse
        f.create_dataset(name,data=encoding)


# In[33]:


App(tkinter.Tk(),"FaceID")


# ____________
