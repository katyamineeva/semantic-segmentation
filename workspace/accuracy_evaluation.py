## note: cv2 works with python2. so type in terminal: alias python=python2 after launch
from __future__ import division
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from namespace import *

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
    
    

def my_accuracy(imgResult = cv2.imread('to_test.tif'), imgIdeal = cv2.imread('to_test1.tif')):
    try:
        cv2.imwrite("my_seg.tif", imgResult)
    except:
        pass
    
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

        TP = sum([getTP(obj, comporation) for obj in objectTypes])
        TN = sum([getTN(obj, comporation) for obj in objectTypes])
        FP = sum([getFP(obj, comporation) for obj in objectTypes])
        FN = sum([getFN(obj, comporation) for obj in objectTypes])

        accuracy = (TP + TN) / (TP + FP + TN + FN)    
        print "\n accuracy", accuracy, "\n"
            

def compute_accuracy(test_set):
    average_accuracy = 0.0

    for num in test_set:
        result_image = get_result_image(num)
        gt_image = get_gt_image(num)

        cur_accuracy = np.mean(result_image == gt_image)
        average_accuracy += cur_accuracy
        print "accuracy of segmentation picture ", num, " = ", cur_accuracy
    average_accuracy /= len(test_set)
    print "avarage accuracy = ", average_accuracy



#accuracy_1()   
#cv2.waitKey()
