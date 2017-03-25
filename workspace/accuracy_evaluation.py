## note: cv2 works with python2. so type in terminal: alias python=python2 after launch
from __future__ import division
import cv2
import numpy as np


IMP_SURF = (255, 255, 255)
BUILDING = (0, 0, 255)
LOW_VEG = (0, 255, 255)
TREE = (0, 255, 0)
CAR = (255, 255, 0)
CLUTTER = (255, 0, 0)

objectTypes = [IMP_SURF, BUILDING, LOW_VEG, TREE, CAR, CLUTTER]
objectNames = {IMP_SURF : "IMP_SURF", BUILDING : "BUILDING", LOW_VEG : "LOW_VEG", TREE : "TREE", CAR : "CAR", CLUTTER : "CLUTTER"}


def equiv(a, b):
    return ((a[0] == b[0]) and (a[1] == b[1]) and (a[2] == b[2]))

def getTP(obj, comporation):
    return comporation[obj][obj]

def getFP(obj, comporation):
    notObj = objectTypes[:]
    notObj.remove(obj)
    res = 0
    for anotherObj in notObj:
        res += comporation[obj][anotherObj]
    return res


def getFN(obj, comporation):
    notObj = objectTypes[:]
    notObj.remove(obj)
    res = 0
    for anotherObj in notObj:
        res += comporation[anotherObj][obj]
    return res


def getTN(obj, comporation):
    notObj = objectTypes[:]
    notObj.remove(obj)
    res = 0
    for anotherObj in notObj:
        for oneMoreObj in notObj:
            res += comporation[anotherObj][oneMoreObj]
    return res
    
    

def accuracy(imgResult = cv2.imread('to_test.tif'), imgIdeal = cv2.imread('to_test1.tif')):
    
    detectedOnImage = dict.fromkeys(objectTypes, False)


    if (imgResult.shape != imgIdeal.shape):
        print "sizes of images are distinct" 
    else:
        high = imgResult.shape[0]
        width = imgResult.shape[1]

        emptyDict = dict.fromkeys(objectTypes, 0)

        comporation = dict.fromkeys(objectTypes, {})
        for obj in objectTypes:
            comporation[obj] = emptyDict.copy()

        precision = emptyDict.copy()
        recall = emptyDict.copy()
        accuracy = emptyDict.copy()
        f1 = emptyDict.copy()
    
        for y in range(high):
            for x in range(width):
                pixelResult = tuple(imgResult[y][x])
                pixelIdeal = tuple(imgIdeal[y][x])

                detectedOnImage[pixelIdeal] = True
                detectedOnImage[pixelResult] = True

                comporation[pixelResult][pixelIdeal] += 1


    
        for obj in objectTypes:

            if not detectedOnImage[obj]:
                print "no", objectNames[obj], "is detected \n"
                continue

            print "objectType: ", objectNames[obj]
            tp = getTP(obj, comporation)
            tn = getTN(obj, comporation)

            fp = getFP(obj, comporation)
            fn = getFN(obj, comporation)
            if (tp + fp) > 0:
                precision[obj] = tp / (tp + fp)
                print "precision", precision[obj]
            if (tp + fn) > 0:
                recall[obj]  = tp / (tp + fn)
                print "recall", recall[obj]
                

            if  (precision[obj] + recall[obj]) > 0:
                f1[obj] = (2 * precision[obj] * recall[obj]) / (precision[obj] + recall[obj])
                print "f1", f1[obj]
            accuracy[obj] = (tp + tn) / (tp + tn + fp + fn)    
            print "accuracy", accuracy[obj], "\n"
            


#accuracy()
    
cv2.waitKey()
