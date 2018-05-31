import cv2 as cv
import numpy as np

#Reading the video
cap = cv.VideoCapture('F:\Vehicle Counting\Codes\Chandigarh dataset\second.mp4')
cv.namedWindow('image', cv.WINDOW_NORMAL)


#Functions Used
fgbg=cv.bgsegm.createBackgroundSubtractorMOG()
orb=cv.ORB_create()
bf = cv.BFMatcher()


#Writing the video
#out = cv.VideoWriter('SurfDetectPoints.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (352,240))


#Feature Tracking Masks
line=np.zeros((240,352,3),np.uint8)


#Blob Detector Function
def blobs(f_gray):
    fgmask = fgbg.apply(f_gray)
    blur = cv.GaussianBlur(fgmask, (5, 5), 0)
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2)); kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4,4))
    erosion = cv.morphologyEx(blur, cv.MORPH_ERODE, kernel1)
    dilation = cv.morphologyEx(erosion, cv.MORPH_DILATE, kernel2)
    retu, threshold = cv.threshold(dilation, 50, 255, cv.THRESH_BINARY)
    return(threshold)

#Capturing Frame by Frame
e=0; match_kp=[]; match_des=[]
frame_count = 0

while (1):
    ret, f = cap.read();
    f = cv.resize(f, (640, 480))
    cv.imshow('frame1', f)
    frame_count += 1
    if ret == False:
        break
    if frame_count % 2 == 0:
        continue
    f_gray= cv.cvtColor(f,cv.COLOR_BGR2GRAY)

    #Calling Blob formation function
    blob=blobs(f_gray)

    #Extracting Foreground Objects
    f_gray=np.uint8(blob*f_gray)

    #Drawing Contours of Blobs
    im, contours, hierarchy = cv.findContours(blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        x,y,w,h=cv.boundingRect(cnt)
        if w<10 and h<10:continue
        cv.rectangle(f,(x, y),(x + w, y + h),(200, 0, 0), 2)

    #Extracting SURF Features of Current Frame(only from foreground regions)
    kp, des = orb.detectAndCompute(f_gray, None)
    img2 = cv.drawKeypoints(f_gray, kp, None, (255, 0, 0), 4)
    #out.write(img2)
    cv.imshow('frame', img2)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()