import cv2
import numpy as np
import os


res1 = 0.0


def g1():
    list=[]
    fingerprint_test = cv2.imread("William.BMP")
    # cv2.imshow("Original", cv2.resize(fingerprint_test, None, fx=1, fy=1))
    for file in [file for file in os.listdir("dataset")]:
        fingerprint_database_image = cv2.imread("./dataset/"+file)
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(fingerprint_test, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)
        #print(descriptors_2)
        matches = cv2.FlannBasedMatcher(dict(algorithm=2, trees=10),
                                    dict()).knnMatch(descriptors_1, descriptors_2, k=2)
        match_points = []

        for p, q in matches:
            #print('p',p.distance)
            #print('q',q.distance)
            if p.distance < 0.1*q.distance:

               match_points.append(p)
               #print('match',match_points)
        keypoints = 0
        if len(keypoints_1) <= len(keypoints_2):
           keypoints = len(keypoints_1)
        else:
           keypoints = len(keypoints_2)
        print('key',keypoints)
        print('lenma',len(match_points))


    if (len(match_points) / keypoints)>0.4:
          print("% match: ", len(match_points) / keypoints * 100)
          print("Figerprint ID: " + str(file))
          global res1
          res1 = len(match_points) / keypoints * 100
          result = cv2.drawMatches(fingerprint_test, keypoints_1, fingerprint_database_image,
                               keypoints_2,match_points,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,outImg=None)
          img = cv2.drawKeypoints(fingerprint_test, keypoints_1,None)
          img1=cv2.drawKeypoints(fingerprint_database_image,keypoints_2,None)


          Keypointimage = np.concatenate((img, img1), axis=1)
          Comparedimage=np.concatenate((fingerprint_test,fingerprint_database_image),axis=1)
          Keypointimage = cv2.resize(Keypointimage, None, fx=3, fy=3)
          Comparedimage=cv2.resize(Comparedimage,None,fx=3,fy=3)

          cv2.imshow('Marked keypoints',Keypointimage)
          cv2.imshow('Comparedimage',Comparedimage)


          result = cv2.resize(result, None, fx=1.5, fy=1.5)
          #list = [result, Keypointimage, Comparedimage]
    else:

          fingerprint_test=cv2.putText(fingerprint_test,"Doesn't match any available fingerprint", (0, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
          result = cv2.resize(fingerprint_test, None, fx=1.5, fy=1.5)
          #list=[result]

    return result, res1


if __name__ == "__main__":
    g1()
