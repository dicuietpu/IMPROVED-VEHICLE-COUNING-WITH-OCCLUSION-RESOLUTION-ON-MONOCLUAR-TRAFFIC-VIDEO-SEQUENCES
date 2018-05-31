import cv2 as cv
import numpy as np
import scipy.cluster.hierarchy as hcluster
import math
from matplotlib import pyplot as plt
import itertools

# Reading the video
cap = cv.VideoCapture('F:\Vehicle Counting\Codes\Chandigarh dataset\\second.mp4', cv.IMREAD_GRAYSCALE)

# Writing the video
# out = cv.VideoWriter('Results/Detection6.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30, (868*3,614))


vehicle_count_left = 0
vehicle_count_right = 0
REF_X = 640
REF_Y = 180
# bg +. subtractor
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
bf = cv.BFMatcher()
# Feature detector
# fast = cv.FastFeatureDetector_create()
# fast = cv.FastFeatureDetector_create(50)
orb = cv.ORB_create()
newkp = 0
frame_count = 0
keypoints_list = []
match_descriptors_list = []
match_keypoints_list = []
good_new = []
good_old = []
line = np.zeros((480, 640, 3), np.uint8)

while (1):
    des_list = []
    kp_list = []
    ret, frame = cap.read()
    frame_count += 1
    frame = cv.resize(frame, (640, 480))
    if ret == False:
        break
    if frame_count % 2 == 0:
        continue

    # bg subtraction -> feature detection

    fgmask = fgbg.apply(frame)
    blurred_image = cv.GaussianBlur(fgmask, (5, 5), 0)
    # Applying Surf
    kp = orb.detect(blurred_image, None)
    kp, des = orb.compute(blurred_image, kp)
    # kp = fast.detect(blurred_image, None)
    #print ("k 1 =",kp)
    #print ("des 1", des)
    frame_with_features = cv.drawKeypoints(blurred_image, kp, None, color=(255, 255, 0))
    frame_with_bb = np.copy(frame)
    # print(des)

    if len(kp) < 2:
        continue

    kp_coordinates = []
    temp = []

    for i in range(len(kp)):
        temp.append(kp[i])
    for i in range(len(temp)):
        kp_coordinates.append(np.asarray(temp[i].pt))
    kp_coordinates = np.asarray(kp_coordinates)
    kp_coordinates = np.float32(kp_coordinates)

#-------------------------------------------------Hierarichal Clustering--------------------------------------------------------
    thresh = 14
    clusters = hcluster.fclusterdata(kp_coordinates, thresh, criterion="distance")

    clusters = clusters - 1

    cluster_counter = np.zeros([len(set(clusters))], dtype=int)
    cluster_points = np.empty([len(set(clusters)), len(kp_coordinates), 2])
    cluster_points[:, :, :] = np.nan

    for i in range(len(clusters)):
        cluster_points[clusters[i], cluster_counter[clusters[i]], 0] = kp_coordinates[i, 0]
        cluster_points[clusters[i], cluster_counter[clusters[i]], 1] = kp_coordinates[i, 1]
        cluster_counter[clusters[i]] += 1



    for i in range(len(set(clusters))):
        cluster_list = cluster_points[i, :, :]
        cluster_list = cluster_list[~np.isnan(cluster_list)]
        clean_cluster_list = np.reshape(cluster_list, (-1, 2))



        x_min = int(np.nanmin(cluster_points[i, :, 0]))
        x_max = int(np.nanmax(cluster_points[i, :, 0]))
        y_min = int(np.nanmin(cluster_points[i, :, 1]))
        y_max = int(np.nanmax(cluster_points[i, :, 1]))


#-------------------------------------Calculating point closest to centroid (median)-----------------------------------------------------
        if x_max - x_min > 50 or y_max - y_min > 50:
            # Drawing a bounding rectangle
            cv.rectangle(frame_with_bb, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            # Finding centroid of the clusters
            center = (int((x_max + x_min) / 2), int((y_max + y_min) / 2))
            first_dist = np.sqrt((center[0] - clean_cluster_list[0, 0]) ** 2 + (center[1] - clean_cluster_list[1, 1]) ** 2)
            # Calculating point closest to centroid
            for j in range(len((clean_cluster_list))):
                dist = np.sqrt((center[0] - clean_cluster_list[j, 0]) ** 2 + (center[1] - clean_cluster_list[j, 1]) ** 2)
                if (dist <= first_dist):
                    first_dist = dist
                    #Storing the point (median)
                    median = clean_cluster_list[j, :]
                    median = tuple(median)
                    print ("Cluster", i)
                    print ("Median for this cluster=", median )
                    #Finding the descriptor of that point (median point)
                    for k in range(len(kp)):
                        if (median == kp[k].pt):
                            index = k
                            #newdes = des[index]
                            newkp = kp[k]

                #des_list.append(newdes)
                if (newkp == 0):
                    continue
                kp_list.append(newkp)
                cv.circle(frame_with_bb, (center), 1, (0, 0, 255), -1)

    # #Making list of keypoints
    keypoints_list = kp_list
    # Calculating descriptors from these keypoints
    keypoints_list, desc = orb.compute(blurred_image, keypoints_list)




    # -------------------------------------------------Tracking the centroid----------------------------------------------------



    if frame_count % 2 != 0:
        if len(match_keypoints_list) < 2:
            match_keypoints_list = keypoints_list
            match_desc = desc
            frame_with_bb = cv.add(frame_with_bb, line)



        else:
            # Matching centroids in current frame and matched centroids in previous frame
            matches = bf.knnMatch(desc, match_desc, k=2)
            # matches = bf.knnMatch(np.asarray(centroid_list, np.float32), np.asarray(match_centroid_list, np.float32), k=2)
            #print ("matches", matches)

            # Lowe's Ratio Test
            lowe = []
            for m, n in matches:
                if m.distance < n.distance:
                    lowe.append(m)

            # Finding (x,y) coordinates of matched keypoints
            list_querypt = np.float32([keypoints_list[mat.queryIdx].pt for mat in lowe])
            list_trainpt = np.float32([match_keypoints_list[mat.trainIdx].pt for mat in lowe])

            # Findind good matches based on displacement constraint
            good = []
            for i in range(0, len(lowe)):
                dist = np.linalg.norm(list_querypt[i] - list_trainpt[i])
                if dist > 2 and dist < 30:
                    good.append(lowe[i])

            #print ("good =", good)

            # Drawing tracks of features
            good_new = np.float32([keypoints_list[mat.queryIdx].pt for mat in good])
            #print ("first good new", good_new)
            good_old = np.float32([match_keypoints_list[mat.trainIdx].pt for mat in good])
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                #print (a,b,c,d)
                line = cv.line(line, (a, b), (c, d), (250, 0, 150), 2)
            frame_with_bb = cv.add(frame_with_bb, line)

            # Updating match_kp and match_des
            match_keypoints_list = [keypoints_list[m.queryIdx] for m in good]
            match_desc = [desc[m.queryIdx] for m in good]

            # Adding new features every 9th frame
            if frame_count % 9 == 0:
                for i, element in enumerate(keypoints_list):
                    if element not in match_keypoints_list:
                        match_keypoints_list.append(element)
                        match_desc.append(desc[i])

            match_desc = np.array(match_desc)



    else:
        frame_with_bb = cv.add(frame_with_bb, line)




# -----------------------------------------------------------------------------Counting no of vehicles-----------------------------------------------------------------------------


    #Removing duplicates in good_new
    b_set = set(tuple(x) for x in good_new)
    new_good_new = [list(x) for x in b_set]

    #print ("new ", new_good_new)

    # Counting the vehicles
    for i in range(len(new_good_new)):
        a = new_good_new[i][0]
        b = new_good_new[i][1]
        #c, d = old.ravel()
        #print ("a, b", a,b)
        if b < REF_Y and b > REF_Y - 5:
            #print ("b =", b)
            #print ("yes")
            #print(a)
            if a < 300:
                vehicle_count_left += 1
            if a > 300:
                vehicle_count_right += 1

    '''cv.line(frame_with_bb, (0, REF_Y), (REF_X, REF_Y), (0, 255, 0), 2)


    font = cv.FONT_HERSHEY_SIMPLEX
    cv.rectangle(frame_with_bb, (4, 20), (50, 55), (0, 0, 0), -1)
    cv.rectangle(frame_with_bb, (558, 20), (613, 55), (0, 0, 0), -1)
    cv.putText(frame_with_bb, str(vehicle_count_left), (10, 50), font, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame_with_bb, str(vehicle_count_right), (570, 50), font, 0.8, (0, 255, 255), 2, cv.LINE_AA)'''












    cv.imshow('image', frame_with_bb)

    k = cv.waitKey(30) & 0xFF

    if k == 27:
        break

cap.release()
# out.release()







