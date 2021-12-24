import cv2
from random import randrange


trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#We will be using the laptops webcam

#device webcam. this will connect code to device camara.
webcam = cv2.VideoCapture(0)

#Iterate forever over frames
while True: # this runs forever until we end the webcam video
     successful_frame_read, frame = webcam.read() # video 1:12

    #must convert to grayscale to allow it work. then convert to colour for AI to detect face
     grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     # detect faces     this is looking for the face composition. whether it gets small or big. it will detect it
     face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

     # draw rectangles around faces + do a for loop. saves having to repeat code.
     for (x, y, w, h) in face_coordinates:
          # (x, y, w, h) = face_coordinates[0]
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)


     cv2.imshow('Clever Programmer Face Detector', frame)
     cv2.waitKey(1) # this now runs none stop like a video webcam until a button is clicked

     print("Code compelted.")