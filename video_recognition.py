'''
This script applys a trained model to detect guns in a video clip and saved the detection video
'''

import numpy as np
import cv2
import imutils
import datetime

# cascade
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# input video
input_video = cv2.VideoCapture('data/input.webm')

# save video - Set resolutions, convert them from float to integer. 
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps= 60.0
frame_width = int(input_video.get(3)) 
frame_height = int(input_video.get(4))
size = (frame_width, frame_height)
out = cv2.VideoWriter('output.avi', fourcc, fps, size) 

# out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,360))

# initialize the first frame in the video stream
firstFrame = None

# flag if any gun exists
gun_exist = False

# check whether input video is opened previously or not 
if not input_video.isOpened():  
    print("Error reading video file") 

# loop over the frames of the video    
while True:
    (grabbed, frame) = input_video.read()

    # if the frame could not be grabbed, then we have reached the end of the video
    if not grabbed:
        break
            
    # resize the frame, convert it to grayscale, and blur it
    # frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # detect & rect gun
    gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize = (100, 100))    
    if not gun_exist and len(gun)> 0:
        gun_exist = True        
    for (x,y,w,h) in gun:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]    

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # draw the text and timestamp on the frame
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    
    # write the output video
    out.write(frame)

    # show the frame
    cv2.imshow("Security Feed", frame)    
    
    
    # quit by 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# finish detection
if gun_exist:
    print("guns detected")
else:
    print("guns NOT detected")

# cleanup the input_video, output stream, and close any open windows
input_video.release()
out.release()
cv2.destroyAllWindows()






