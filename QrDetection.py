"""
https://askubuntu.com/questions/348838/how-to-check-available-webcams-from-the-command-line
"""

import cv2
# initalize the cam
#cap = cv2.VideoCapture(2)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap = cv2.VideoCapture(2)

#codec = 0x47504A4D  # MJPG
#cap.set(cv2.CAP_PROP_FPS, 30.0)
#cap.set(cv2.CAP_PROP_FOURCC, codec)
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

#cap.set(3, 1280)
#cap.set(4, 720)
# initialize the cv2 QRCode detector
detector = cv2.QRCodeDetector()
while True:
    _, img = cap.read()
    print ("Resolucion:",img.shape)
    # detect and decode
    data, bbox, _ = detector.detectAndDecode(img)
    # check if there is a QRCode in the image
    if bbox is not None:
        # display the image with lines
        #for i in range(len(bbox)):
            # draw all lines
            #print (bbox)
            #cv2.line(img, tuple(bbox[i][0]), tuple(bbox[(i+1) % len(bbox)][0]), color=(255, 0, 0), thickness=2)
        if data:
            print (bbox)
            print (bbox[0])
            print (bbox[0][0])
            print (bbox[0][1])
            print (bbox[0][2])
            print (bbox[0][3])
            #print (bbox[i+1][0])
            img = cv2.line(img,(int(bbox[0][0][0]),int(bbox[0][0][1])),(int(bbox[0][1][0]),int(bbox[0][1][1])),(0,0,255),3)
            img = cv2.line(img,(int(bbox[0][1][0]),int(bbox[0][1][1])),(int(bbox[0][2][0]),int(bbox[0][2][1])),(0,0,255),3)
            img = cv2.line(img,(int(bbox[0][2][0]),int(bbox[0][2][1])),(int(bbox[0][3][0]),int(bbox[0][3][1])),(0,0,255),3)
            img = cv2.line(img,(int(bbox[0][3][0]),int(bbox[0][3][1])),(int(bbox[0][0][0]),int(bbox[0][0][1])),(0,0,255),3)


            print("[+] QR Code detected, data:", data)
    # display the result
    cv2.imshow("img", img)    
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
