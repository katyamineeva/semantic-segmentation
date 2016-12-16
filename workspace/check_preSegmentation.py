from slic import preSegmentation 
from accuracy_evaluation import accuracy

IMP_SURF = (255, 255, 255)
BUILDING = (0, 0, 255)
LOW_VEG = (0, 255, 255)
TREE = (0, 255, 0)
CAR = (255, 255, 0)
CLUTTER = (255, 0, 0)

objectTypes = [IMP_SURF, BUILDING, LOW_VEG, TREE, CAR, CLUTTER]



def classifySegment(segment, imgNew, imgIdeal):
    cntOfClasses = dict.fromkeys(objectTypes, 0)
    for pixel in segment:
        typeOfPixel = imgIdeal[pixel[0]][pixel[1]]
        cntOfClasses[typeOfPixel] += 1
    
    majorType = TREE
    for obj in objectTypes:
        if cntOfClasses[obj] > cntOfClasses[majorType]:
            majorType = obj
    
    for pixel in segment:
        imgNew[pixel[0]][pixel[1]] = majorType



def preSegmentation_accuracy(filename = "top1.tif"):

    imgIdeal = cv2.imread('gt1.tif')
    imgNew = imgIdeal[:]
    
    segments = preSegmentation(filename)
    for segment in segments:
        classifySegment(segment, imgNew, imgIdeal)
    accuracy(imgNew, imgIdeal)

