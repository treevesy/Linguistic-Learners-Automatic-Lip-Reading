import pickle
from django.conf import settings
from cam_app import views
from django.http import StreamingHttpResponse
from django.http import HttpResponse
import sqlite3
import datetime

# import some common libraries
import numpy as np
import os, json, cv2, random, glob, uuid
import matplotlib.pyplot as plt

from pathlib import Path
import time
from datetime import datetime

video_directory = 'word_videos/'
video_name = "word"
video_extension = ".mov"

class VideoCamera(object):
    #frame_count = 0
    #video_splits = 0
    #current_video_frames = []

    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        self.current_video_frames = []

    def __del__(self):
        self.video.release()

    def get_frame_with_detection(self):
        success, image = self.video.read()
        # replace lines 33-53 with the code needed for your model and then set your output image
        #  you can also include functiond from database operations into camera
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        outputs = image
        # if you dont want to show the detection, comment the below code till outputImage = image, and change it to outputImage = outputs
       
        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor("../lipreading_model/shape_predictor_68_face_landmarks.dat")

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # faces = detector(gray)

        # for face in faces:
        #     x1 = face.left()
        #     y1 = face.top()
        #     x2 = face.right()
        #     y2 = face.bottom()
        #     #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        #     landmarks = predictor(gray, face)
        #     for n in range(0, 68):
        #         x = landmarks.part(n).x
        #         y = landmarks.part(n).y
        #         cv2.circle(image, (x, y), 4, (255, 0, 0), -1)

        # outputImage = image
        # ret, outputImagetoReturn = cv2.imencode('.jpg', outputImage) # check if it work
        # return outputImagetoReturn.tobytes(), outputImage
        


        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # gray = cv2.equalizeHist(gray)
        # #-- Detect faces
        # faces = face_cascade.detectMultiScale(gray)
        # for (x,y,w,h) in faces:
        #     center = (x + w//2, y + h//2)
        #     image = cv2.ellipse(image, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        #     faceROI = gray[y:y+h,x:x+w]
        #     #-- In each face, detect eyes
        #     eyes = eyes_cascade.detectMultiScale(faceROI)
        #     for (x2,y2,w2,h2) in eyes:
        #         eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #         radius = int(round((w2 + h2)*0.25))
        #         image = cv2.circle(image, eye_center, radius, (255, 0, 0 ), 4)
        outputImage = image
        ret, outputImagetoReturn = cv2.imencode('.jpg', outputImage) # check if it work

        #frame_count += 1
        #current_video_frames.append(image)
        self.current_video_frames.append(image)
        print(len(self.current_video_frames))
        return outputImagetoReturn.tobytes(), outputImage

    def createVideoSnippet(self, response):
        # datetime object containing current date and time
        now = datetime.now()
        datetime_clicked = now.strftime("%d-%m-%Y_%H-%M-%S")
        print("Video snippet button clicked at time: ", datetime_clicked)

        full_video_name = video_directory + video_name + '-' + datetime_clicked + video_extension
        copied_frames = self.current_video_frames.copy()
        self.current_video_frames = []
        
        width  = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        fps =  self.video.get(cv2.CAP_PROP_FPS)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(full_video_name, fourcc, fps, (width,height))

        for frame in copied_frames:
            video.write(frame)

        video.release()

        return HttpResponse(json.dumps({'videoName': full_video_name}))


def generate_frames(camera):
    try:
        while True:
            frame, img = camera.get_frame_with_detection()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print(e)

    finally:
        print("Reached finally, detection stopped")
        cv2.destroyAllWindows()
