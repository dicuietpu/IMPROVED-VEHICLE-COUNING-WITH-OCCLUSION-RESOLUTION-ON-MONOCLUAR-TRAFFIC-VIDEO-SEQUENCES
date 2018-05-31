import cv2 as cv
import numpy as np
import time



REF_X =640
REF_Y = 270
#Reading the video
cap = cv.VideoCapture('F:\Vehicle Counting\Codes\Chandigarh dataset\\second.mp4')
cv.namedWindow('image', cv.WINDOW_NORMAL)


#Functions Used
fgbg=cv.bgsegm.createBackgroundSubtractorMOG()
surf=cv.xfeatures2d.SURF_create()
bf = cv.BFMatcher()


#Writing the video
#out = cv.VideoWriter('TrackingUsingSurf.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (352,240))


#Feature Tracking Masks
line=np.zeros((480,640,3),np.uint8)


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
frame_count=0
while (1):
    ret, f = cap.read(); e+=1
    f=cv.resize(f, (640,480))
    if ret == False: break
    if e%2 == 0:
        continue

    f_gray= cv.cvtColor(f,cv.COLOR_BGR2GRAY)

    #Calling Blob formation function
    blob=blobs(f_gray)

    #Extracting Foreground Objects
    f_gray=np.uint8(blob*f_gray)

    #Drawing Contours of Blobs
    im, contours, hierarchy = cv.findContours(blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv.contourArea(cnt)<300:
            continue
        x,y,w,h=cv.boundingRect(cnt)
        if w<10 and h<10:continue
        cv.rectangle(f,(x, y),(x + w, y + h),(0, 200, 0), 2)

    #Extracting SURF Features of Current Frame(only from foreground regions)
    kp,des= surf.detectAndCompute(f_gray, None)





    if e % 2 != 0:
        if len(match_kp) < 2:
            match_kp = kp
            match_des = des
            f = cv.add(f, line)

        else:

            # Matching Features in current frame keypoints and matched keypoints from previous frame
            matches = bf.knnMatch(des, match_des, k=2)

            # Lowes's Ratio Test
            lowe = []
            for m, n in matches:
                if m.distance < n.distance:
                    lowe.append(m)

            # Finding (x,y) coordinates of matched keypoints
            list_querypt = np.float32([kp[mat.queryIdx].pt for mat in lowe])
            list_trainpt = np.float32([match_kp[mat.trainIdx].pt for mat in lowe])

            # Finding good matches based on displacement contraint
            good = []
            for i in range(0, len(lowe)):
                dist = np.linalg.norm(list_querypt[i] - list_trainpt[i])
                if dist > 2 and dist < 30:
                    good.append(lowe[i])

            # Drawing tracks of features
            good_new = np.float32([kp[mat.queryIdx].pt for mat in good])
            good_old = np.float32([match_kp[mat.trainIdx].pt for mat in good])
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                line = cv.line(line, (a, b), (c, d), (250, 0, 150), 1)
            f = cv.add(f, line)
            print (f)

            # Updating match_kp and match_des
            match_kp = [kp[m.queryIdx] for m in good]
            match_des = [des[m.queryIdx] for m in good]

            # Adding new features every 9th frame
            if e % 9 == 0:
                for i, element in enumerate(kp):
                    if element not in match_kp:
                        match_kp.append(element)
                        match_des.append(des[i])

            #Converting match_des to numpy array
            match_des = np.array(match_des)

            # Pruning out the lines
            '''for i in range(len(kp) -1,  -1, -1):
                if frame_time - match_kp[i]['last_seen'] > 0.7:
                    del kp[i]'''

    else:
        f = cv.add(f, line)

    '''cv.line(f, (0, REF_Y), (REF_X, REF_Y), (0, 255, 0), 2)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.rectangle(f, (4, 20), (50, 55), (0, 0, 0), -1)
    cv.rectangle(f, (558, 20), (613, 55), (0, 0, 0), -1)
    cv.putText(f, str(vehicle_count_left), (10, 50), font, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(f, str(vehicle_count_right), (570, 50), font, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    # out.write(frame)
    #cv.imshow('Detection', frame)
    # cv.imshow('Blob', blob)'''






    #Output
    #out.write(f)
    cv.imshow('image',f)
    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break

#out.release()