import cv2
import numpy as np


cap = cv2.VideoCapture('Crowd_Video_Stock_Footage.mp4')
ret , frame = cap.read()
face_case = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_rects = face_case.detectMultiScale(frame)

x,y,w,h = tuple(face_rects[0])
track_window = (x,y,w,h)

roi = frame[y:y+h,x:x+w]
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv],[0],None,[180],[0,180])
roi_hist = cv2.normalize(roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 2)
while(True):
	ret,frame = cap.read()
	if ret == True:
		hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		dest_roi = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

		"""
		###########               MEANSHIFT                #############
		_, track_window = cv2.meanShift(dest_roi,track_window,term_crit)

		x,y,w,h = track_window

		img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),3)
		
		"""

		###########               CAMSHIFT                ##############

		rect , _ = cv2.CamShift(dest_roi,track_window,term_crit)

		pts=cv2.boxPoints(rect)
		pts=np.int0(pts)

		img2 = cv2.polylines(frame,[pts],True,(0,0,0),3)


		###########      
		cv2.imshow('Result',img2)


		if cv2.waitKey(50) and 0xFF == 27:
			break

	else:
		break


cap.release()
cv2.destroyAllWindows()
