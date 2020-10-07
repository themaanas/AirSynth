import cv2
import numpy as np
import copy
import math
import time
import pygame, sys
from pygame.locals import *
from pygame import mixer
import copy

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 40  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
isInKey = False
tiles = pygame.image.load('pianoTile.png')
mixer.init()

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
screen = pygame.display.set_mode([1280,720])

cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
point_list = []
polygon_list = []
is_drawing = False

try:
    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        # frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                     (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

        contour_frame = np.zeros((1280,720,3), np.uint8)
        # contour_frame = removeBG(contour_frame)


        # cv2.imshow('original', frame)
        #  Main operation
        if isBgCaptured == 1:  # this part wont run until background captured
            img = removeBG(frame)
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
            cv2.imshow('mask', img)

            #convert the image into binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            # cv2.imshow('blur', blur)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            # cv2.imshow('ori', thresh)


            # get the contours
            thresh1 = copy.deepcopy(thresh)
            contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i

                res = contours[ci]
                hull = cv2.convexHull(res)
                drawing = np.zeros(img.shape, np.uint8)
                cv2.drawContours(contour_frame, [res], 0, (0, 255, 0), 2)
                c = max(contours, key=cv2.contourArea)
                # x, y = str(hull[0][0])[1:-1].split()
                # print([hull])
                cv2.drawContours(contour_frame, [hull], 0, (0, 0, 255), 3)
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
                isFinishCal,cnt = calculateFingers(res,frame)
                print(cnt)
                cv2.circle(frame, extTop, 8, (255, 255, 255), -1)
                x,y = extTop
                if (y < 190 and not isInKey):
                    isInKey = True
                    if (x > 5 and x < 95):
                        print("C")
                        mixer.music.load("C.mp3")
                        mixer.music.play()
                    elif(x > 95 and x < 185):
                        print("D")
                        mixer.music.load("D.mp3")
                        mixer.music.play()
                    elif(x > 185 and x < 275):
                        print("e")
                        mixer.music.load("E.mp3")
                        mixer.music.play()
                    elif(x > 275 and x < 365):
                        print("f")
                        mixer.music.load("F.mp3")
                        mixer.music.play()
                    elif(x > 365 and x < 455):
                        print("g")
                        mixer.music.load("G.mp3")
                        mixer.music.play()
                    elif(x > 455 and x < 545):
                        print("a")
                        mixer.music.load("A.mp3")
                        mixer.music.play()
                    elif(x > 545 and x < 635):
                        print("b")
                        mixer.music.load("B.mp3")
                        mixer.music.play()
                if (y > 190):
                    isInKey = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.swapaxes(frame, 0,1)
        frame = pygame.surfarray.make_surface(frame)
        contour_frame = np.swapaxes(contour_frame, 0,1)
        contour_frame = pygame.surfarray.make_surface(contour_frame)
        contour_frame.set_colorkey((0,0,0))
        contour_frame.set_alpha(100)

        tile_length = 190

        pygame.draw.line(frame, (255,255,255), (5,0),(5,tile_length))
        pygame.draw.line(frame, (255,255,255), (95,0),(95,tile_length))
        pygame.draw.line(frame, (255,255,255), (185,0),(185,tile_length))
        pygame.draw.line(frame, (255,255,255), (275,0),(275,tile_length))
        pygame.draw.line(frame, (255,255,255), (365,0),(365,tile_length))
        pygame.draw.line(frame, (255,255,255), (455,0),(455,tile_length))
        pygame.draw.line(frame, (255,255,255), (545,0),(545,tile_length))
        pygame.draw.line(frame, (255,255,255), (635,0),(635,tile_length))




        screen.blit(frame, (0,0))
        screen.blit(tiles, (0,0))
        screen.blit(contour_frame, (0,0))




        pygame.display.update()
        for event in pygame.event.get():
            if event.type == KEYUP and event.key == pygame.K_ESCAPE:
                camera.release()
                pygame.quit()
                cv2.destroyAllWindows()
            elif event.type == KEYUP and event.key == pygame.K_b:
                bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
                isBgCaptured = 1
                print( '!!!Background Captured!!!')
except SystemExit:
    pygame.quit()
    cv2.destroyAllWindows()
