import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from accuracy_evaluation import accuracy
import numpy as np

IMP_SURF = (255, 255, 255)
BUILDING = (0, 0, 255)
LOW_VEG = (0, 255, 255)
TREE = (0, 255, 0)
CAR = (255, 255, 0)
CLUTTER = (255, 0, 0)

objectTypes = [IMP_SURF, BUILDING, LOW_VEG, TREE, CAR, CLUTTER]



def slic_segmentation(filename, display = False, numSegments = 200):
    image = img_as_float(io.imread(filename))
    
    # sigma smoothing Gaussian kernel, unnessesary parameter 
    segments = slic(image, n_segments = numSegments, sigma = 5, convert2lab = True)
    if display:
        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")
        plt.show()

    # returns array with same dimentions as image 
    # but value of array[x][y] -- number of cluster, pixel (x, y) belongs to
    return segments




def equiv(a, b):
    return ((a[0] == b[0]) and (a[1] == b[1]) and (a[2] == b[2]))



def get_list_of_segments(mask, cnt_of_segments, num):
    list_of_segments = [[]  for i in range(cnt_of_segments)]

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            segment_num = mask[i][j]
            list_of_segments[segment_num].append(tuple([i, j]))
    list_of_segments = np.asarray(list_of_segments)
    list_of_segments.tofile("list_of_segments_" + str(num) + ".txt", ", ")
    
    return list_of_segments



def presegmentation(num):
    photo_top = "top" + str(num) + ".tif"
    image_top = cv2.imread(photo_top)

    mask = slic_segmentation(photo_top, numSegments = 300)
    mask.tofile("mask_of_" + str(num) + "_image.txt", ", ")
    cnt_of_segments = max([max(row) for row in mask]) + 1
    return mask



#presegmentation_accuracy()

