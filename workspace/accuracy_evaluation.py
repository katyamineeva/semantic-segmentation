import cv2
import numpy as np

IMP_SURF = (255, 255, 255)
BUILDING = (0, 0, 255)
LOW_VEG = (0, 255, 255)
TREE = (0, 255, 0)
CAR = (255, 255, 0)
CLUTTER = (255, 0, 0)

objectTypes = [IMP_SURF, BUILDING, LOW_VEG, TREE, CAR, CLUTTER]


def equiv(a, b):
    return ((a[0] == b[0]) and (a[1] == b[1]) and (a[2] == b[2]))

def getTP(obj):
    return comporation[obj][obj]

def getFP(obj):
    notObj = objectTypes[:]
    notObj.remove(obj)
    res = 0
    for anotherObj in notObj:
        res += comporation[obj][anotherObj]
    return res


def getFN(obj):
    notObj = objectTypes[:]
    notObj.remove(obj)
    res = 0
    for anotherObj in notObj:
        res += comporation[anotherObj][obj]
    return res


def getTN(obj):
    notObj = objectTypes[:]
    notObj.remove(obj)
    res = 0
    for anotherObj in notObj:
        for oneMoreObj in notObj:
            res += comporation[anotherObj][oneMoreObj]
    return res
    
    

def accuracy(imgResult = cv2.imread('gt1.tif'), imgIdeal = cv2.imread('gt1.tif')):      
    
    if (imgResult.shape != imgIdeal.shape):
        print "sizes of images are distinct"
    else:
        high = imgResult.shape[0]
        width = imgResult.shape[1]
        comporation = dict.fromkeys(objectTypes, dict.fromkeys(objectTypes, 0))

    
        for y in range(high):
            for x in range(width):
                pixelResult = tuple(imgResult[y][x])
                pixelIdeal = tuple(imgIdeal[y][x])
                comporation[pixelResult][pixelIdeal] += 1
    
        precision = dict.fromkeys(objectTypes, 0)
        recall = dict.fromkeys(objectTypes, 0)
        accuracy = dict.fromkeys(objectTypes, 0)
        f1 = dict.fromkeys(objectTypes, 0)
    
        for obj in objectTypes:
            tp = getTP(obj)
            tn = getTN(obj)
            fp = getFP(obj)
            fn = getFN(obj)
    
            precision[obj] = tp / (tp + fp)
            recall[obj]  = tp / (tp + fn)
            accuracy[obj] = (tp + tn) / (tp + tn + fp + fn)
            f1[obj] = (2 * precision[obj] * recall[obj]) / (precision[obj] + recall[obj])
    
            print "objectType: ", obj
            print "precision", precision[obj]
            print "recall", recall[obj]
            print "accuracy", accuracy[obj]
            print "f1", f1[obj], "\n"

    
    
##cv2.waitKey()