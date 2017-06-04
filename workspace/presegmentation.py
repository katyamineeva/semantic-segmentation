from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import os
import cPickle
from namespace import *



def slic_segmentation(filename, numSegments = 300, sig = 5, display = False):
    image = img_as_float(io.imread(filename))
    
    # sigma smoothing Gaussian kernel, unnessesary parameter 
    segments = slic(image, n_segments = numSegments, sigma = sig, convert2lab = True)
    if display:
        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")
        plt.show()

    # returns array with same dimentions as image 
    # but value of array[x][y] -- number of cluster, pixel (x, y) belongs to
    return segments




# list_of_segments -- 2d array
# list_of_segments[i] -- an array of pixel coordinates in segment i

def compute_list_of_segments(mask):
    cnt_of_segments = max([max(row) for row in mask]) + 1
    list_of_segments = [[]  for i in range(cnt_of_segments)]

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            segment_num = mask[i][j]
            list_of_segments[segment_num].append(tuple([i, j]))
    return list_of_segments


# ------------------------------------------------------------
# the only one function, which is used outside this file
# num -- picture num 
def presegmentation(num, num_segments = 300, sigma = 5):
    pathname = os.path.expanduser("~/Pictures/project") + "/superpixels"
    if not os.path.exists(pathname):
        os.mkdir(pathname)

    filename = os.path.expanduser("~/Pictures/project") + "/top/top_mosaic_09cm_area" + str(num) + ".tif"

    mask = slic_segmentation(filename, numSegments = num_segments, sig = sigma)
    filename = os.path.expanduser("~/Pictures/project") + "/superpixels/mask" + str(num)
    cPickle.dump(mask, open(filename, "wb+"))
    
    list_of_segments = compute_list_of_segments(mask)
    filename = os.path.expanduser("~/Pictures/project") + "/superpixels/list_of_segments" + str(num)
    cPickle.dump(list_of_segments, open(filename, "wb+")) 
    
##presegmentation(1)