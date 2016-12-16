## Running from shell by: python slic.py --image image.jpg

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
 

def preSegmentation(filename = "", display = False):
    if filename == "":
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required = True, help = "Path to the image")
        args = vars(ap.parse_args())
         
        # load the image and convert it to a floating point data type
        image = img_as_float(io.imread(args["image"]))
    else:
        image = img_as_float(io.imread(filename))
    
    # it's better to get more superpixels rather than get a mistake 
    numSegments = 300
    # apply SLIC and extract (approximately) the supplied number of segments
    # sigma smoothing Gaussian kernel, unnessesary parameter 
    segments = slic(image, n_segments = numSegments, sigma = 5, convert2lab = True)
    if display:
        # show the output of SLIC
        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")
             
        # show the plots
        plt.show()
    return segments
