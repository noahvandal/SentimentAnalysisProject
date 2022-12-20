#!/usr/bin/env python3
import random
import tensorflow
from tensorflow import keras
from keras.models import model_from_json
import cv2
import numpy as np
import time

class VideoCNN():

    def __init__(self):
        
        # abs_path = "C:/Users/noahv/OneDrive/NDSU Research/Coding Projects/ML 677 Project Testings/sentiment-analysis/"

        # face_dims = (48,48)

        # emotion_score = [3, 3, 3, 10, 0, 7, 5] ## happy = 10, sad = 0, neutral = 5; disgust = angry = fear = 3; surprise = 7

        # filename = abs_path + "model.h5"
        # model = model_from_json(open(abs_path + "model.json", "r").read())
        # model.load_weights(filename)

        # face_haar_cascade = cv2.CascadeClassifier(abs_path + 'haarcascades/haarcascade_frontalface_default.xml')


        pass

    
    def close(self):
        """
        Cleanup any leftover objects like the microphone.
        """
        pass

    def face_finding(self,gray_image):

        abs_path = "C:/Users/noahv/OneDrive/NDSU Research/Coding Projects/ML 677 Project Testings/sentiment-analysis/"
        face_haar_cascade = cv2.CascadeClassifier(abs_path + 'haarcascades/haarcascade_frontalface_default.xml')

        faces = face_haar_cascade.detectMultiScale(gray_image)

        return faces

    
    def face_analysis(self,faces,gray_image):

        avg_emotion_list = [] ## list to hold emotion score of each face in image

        abs_path = "C:/Users/noahv/OneDrive/NDSU Research/Coding Projects/ML 677 Project Testings/sentiment-analysis/"

        filename = abs_path + "model.h5"
        model = model_from_json(open(abs_path + "model.json", "r").read())
        model.load_weights(filename)

        face_dims = (48,48)
        emotion_score = [3, 3, 3, 10, 0, 7, 5] ## happy = 10, sad = 0, neutral = 5; disgust = angry = fear = 3; surprise = 7


        for (x,y,w,h) in faces:  ## parsing through each recognized image in frame
            gray_face = gray_image[x:x+w,y:y+h]
            gray_face = cv2.resize(gray_face, face_dims)
            gray_face = gray_face[None,:,:]

            print('attempting pred')
            pred_ = model.predict(gray_face)
            print(pred_)

            pred_store = pred_

            prediction = np.argmax(pred_)
            
            score = pred_store * emotion_score
            # print(pred_store)
            score = np.sum(score)
            # print(score)

            avg_emotion_list.append(score)
        
        total_score = np.average(avg_emotion_list)

        return total_score
                

    def inference(self):
        """
        Run the actual inference engine.
        """
        self.__init__()

    
        # TODO: implement AudioCNN inference function
        frame_score = 0

        cam = cv2.VideoCapture(0)  ## 0 is default camera channel
        retval, frame = cam.read()
        print(frame.shape)


        if retval != True:  ## making sure that a frame is present
            raise ValueError("Can't read frame")

        gray_image = np.array(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        faces = self.face_finding(gray_image)
        print(faces)

        try:
            frame_score = self.face_analysis(faces,gray_image)
        except:
            print("no inference!!")

            pass

        cv2.imshow('image',gray_image)
        cv2.waitKey(0)
        cam.release()
        cv2.destroyAllWindows()



        # out = random.randint(0,10)
        return frame_score


# test_cnn = VideoCNN()

# print(test_cnn.__init__)

# test_cnn.__init__()
# score = test_cnn.inference()

# print(score)
