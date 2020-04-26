import cv2

#Defining tracker

def ask_for_tracker():
	print('Choose the tracker you want to use.')
	print('Enter 0 for BOOSTING: Based on the same algorithm used to power the machine learning behind Haar cascades (AdaBoost), but like Haar cascades, is over a decade old. This tracker is slow and doesn’t work very well. Interesting only for legacy reasons and comparing other algorithms ')
	print('Enter 1 for MIL: Better accuracy than BOOSTING tracker but does a poor job of reporting failure. ')
	print('Enter 2 for KCF: Kernelized Correlation Filters. Faster than BOOSTING and MIL. Similar to MIL and KCF, does not handle full occlusion well. ')
	print('Enter 3 for TLD: TLD tracker was incredibly prone to false-positives')
	print('Enter 4 for MEDIANFLOW: Does a nice job reporting failures; however, if there is too large of a jump in motion the model will fail. ')
	print('Enter 5 for GOTURN: The only deep learning-based object detector included in OpenCV.')
	print('Enter 6 for MOSSE: Very, very fast. Not as accurate as CSRT or KCF but a good choice if you need pure speed')
	print('Enter 7 for CSRT: Discriminative Correlation Filter (with Channel and Spatial Reliability). Tends to be more accurate than KCF but slightly slower. ')

	choice = input("Please select your tracker: ")

	if choice =='0':
		tracker = cv2.TrackerBoosting_create()

	if choice =='1':
		tracker = cv2.TrackerMIL_create()

	if choice =='2':
		tracker = cv2.TrackerKCF_create()

	if choice =='3':
		tracker = cv2.TrackerTLD_create()

	if choice =='4':
		tracker = cv2.TrackerMedianFlow_create()

	if choice =='5':
		tracker = cv2.TrackerGOTURN_create()

	if choice =='6':
		tracker = cv2.TrackerMOSSE_create()

	if choice =='7':
		tracker = cv2.TrackerCSRT_create()

	return tracker


tracker = ask_for_tracker()

tracker_name = str(tracker).split()[0][1:]

cap =cv2.VideoCapture('test.mp4')

ret,frame=cap.read()

roi = cv2.selectROI(frame)

#Initialise tracker

ret =tracker.init(frame,roi)

while(True):
	ret,frame = cap.read()

	success , roi = tracker.update(frame)

	x,y,w,h = tuple(map(int,roi))

	print(x,y,w,h)
	if success:

		pts1= (x,y)
		pts2 = (x+w,y+h)

		cv2.rectangle(frame,pts2,pts2,(0,0,0),10)

	else:

		cv2.putText(frame,"Failure to track the object",(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(25,125,125),3)
	

	cv2.putText(frame,tracker_name,(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,125),3)

	cv2.imshow(tracker_name,frame)

	#Exit with esc key

	if cv2.waitKey(50) & 0xFF == 27:
		break

cap.release()
cv2.destroyAllWindows()