import cv2 as cv
import numpy as np

#Reading the video
cap = cv.VideoCapture('F:\Vehicle Counting\Videos\Input\\test.mpg')

font = cv.FONT_HERSHEY_SIMPLEX
#Functions Used
fgbg=cv.bgsegm.createBackgroundSubtractorGMG()



#Post Processing
def blobs(f_gray):
    fgmask = fgbg.apply(f_gray)
    blur = cv.GaussianBlur(fgmask, (21, 21), 0)
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2)); kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4,4))
    erosion = cv.morphologyEx(blur, cv.MORPH_ERODE, kernel1)
    dilation = cv.morphologyEx(erosion, cv.MORPH_DILATE, kernel2)
    retu, threshold = cv.threshold(dilation, 50, 255, cv.THRESH_BINARY)
    return(threshold)


frame_count =0
while (1):
    #Cpaturing frame by frame
    ret, frame = cap.read();
    frame_count +=1
    if ret == False: break
    #frame = cv.resize(frame, (640, 480))
    #frame = frame[80:640, 0:640]
    if frame_count % 2 ==0:
        continue
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask2 = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask3 = np.zeros(frame.shape[:2], dtype=np.uint8)
    f_gray= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    #Calling Blob formation function
    blob=blobs(f_gray)

    #Extracting Foreground Objects
    f_gray=np.uint8(blob*f_gray)

    #Finding Contours of Vehicle
    im, contours, hierarchy = cv.findContours(blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # for each contour
    for cnt in contours:
        if cv.contourArea(cnt)<180:
            continue
        print("cnt =", cnt)
        hull = cv.convexHull(cnt)
        cv.drawContours(frame, [cnt], -1, (255, 0, 0,))
        cv.drawContours(frame, [hull], -1, (255, 255, 255))
        #Calculating compactness of vehicle and convex hull
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        crvehicle = (perimeter*perimeter)/area
        areahull = cv.contourArea(hull)
        perimeterhull = cv.arcLength(hull, True)
        crhull = (perimeterhull*perimeterhull)/areahull
        #Calculating compactness ratio
        cr = crhull/crvehicle
        if (cr < 0.75):
            x, y, w, h = cv.boundingRect(cnt)
            cv.putText(frame, 'occlusion', (int(x+w/2),int(y+h/2)), font, 0.5, (355, 255, 255), 2, cv.LINE_AA)
            #cv.imshow("window4", frame)
            print("occlusion")
            #Calculating mask of area between the convex hull and the vehicle
            cv.drawContours(mask, [cnt], -1, (255,0 ,0), -1)
            cv.drawContours(mask2, [hull], -1,(255, 0, 0), -1)
            mask3 = mask2 - mask
            cv.imshow("window", mask)
            cv.imshow("windows", mask2)
            cv.imshow("windows3", mask3)
            #Finding two largest individual area in the mask
            im, contours2, hierarchy = cv.findContours(mask3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            areas = [cv.contourArea(c) for c in contours2]
            print ("areas =", areas)
            max_index = np.argmax(areas)
            print ("first max_index", max_index)
            cnt2 = contours2[max_index]
            print ("first largest area contour=", cnt2)
            areas.remove(areas[max_index])
            max_index2 = np.argmax(areas)
            print ("second max_index", max_index2)
            print("areas =", areas)
            cnt3 = contours2[max_index2]
            print("second largest area contour=", cnt3)
            cv.drawContours(frame, [cnt2], -1, (255, 0, 0), -1)
            cv.drawContours(frame, [cnt3], -1, (255, 255, 0), -1)
            hull2 = cv.convexHull(cnt2)
            print("Convex Hull first largest", hull2)
            hull3 = cv.convexHull(cnt3)
            print("Convex Hull second largest", hull3)
            #print ("hull =", len(hull2))
            #print ("edge =", len(cnt2))
            #print (hull2[0,0,1])
            #Finding interior distance between the convex hull points and the contour of the area
            min = (cnt2[0,0,0] - hull2[0,0,0])**2 + (cnt2[0,0,1] - hull2[0,0,1])**2
            for k in range(len(cnt2)):
                for j in range(len(hull2)):
                    d = (cnt2[k,0,0] - hull2[j,0,0])**2 + (cnt2[k,0,1] - hull2[j,0,1])**2
                    if d <= min:
                        min = d;
                        first_point = cnt2[k,0,:]
                        first_point = tuple(first_point)


            min = (cnt3[0, 0, 0] - hull3[0, 0, 0]) ** 2 + (cnt3[0, 0, 1] - hull3[0, 0, 1]) ** 2
            for k in range(len(cnt3)):
                for j in range(len(hull3)):
                    d = (cnt3[k, 0, 0] - hull3[j, 0, 0]) ** 2 + (cnt3[k, 0, 1] - hull3[j, 0, 1]) ** 2
                    if d <= min:
                        min = d;
                        second_point = cnt3[k,0,:]
                        second_point = tuple(second_point)

            #Drawing line between the two cutting points
            cv.line(frame, first_point, second_point, (65,95,2), 3)
            cv.imshow("window4", frame)




        else:
            print("no occlusion")









    k = cv.waitKey(30) & 0xff
    if k == 27:
        break