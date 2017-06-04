import os
import cPickle
from skimage import io
import numpy as np
import cv2

IMP_SURF = (255, 255, 255)
BUILDING = (0, 0, 255)
LOW_VEG = (0, 255, 255)
TREE = (0, 255, 0)
CAR = (255, 255, 0)
CLUTTER = (255, 0, 0)

object_types = [IMP_SURF, BUILDING, LOW_VEG, TREE, CAR, CLUTTER]


CNT_OF_FEATURES = 0
CNT_OF_CLASSES = len(object_types)


RGB = "RGB"
HSV = "HSV"
CIELab = "CIELab"
DSM = "DSM"
models = [RGB, HSV, CIELab, DSM]




def get_mask(num):
    filename = os.path.expanduser("~/Pictures/project") + "/superpixels/mask" + str(num)
    mask = cPickle.load(open(filename, "rb"))
    return mask


def get_list_of_segments(num):
    filename = os.path.expanduser("~/Pictures/project") + "/superpixels/list_of_segments" + str(num)
    list_of_segments = cPickle.load(open(filename, "rb"))
    return list_of_segments


def get_features_value(num):        
    filename = os.path.expanduser("~/Pictures/project") + "/features/features_value" + str(num)
    features = cPickle.load(open(filename, "rb"))
    return features


def get_features_list_of_dicts(num):        
    filename = os.path.expanduser("~/Pictures/project") + "/features/list_of_dicts" + str(num)
    lists_of_features = cPickle.load(open(filename, "rb"))
    return lists_of_features


def get_segments_classes(num):
    filename = os.path.expanduser("~/Pictures/project") + "/features/segments_classes" + str(num)
    segments_classes = cPickle.load(open(filename, "rb"))
    return segments_classes


def get_top_image(num):
    filename = os.path.expanduser("~/Pictures/project") + "/top/top_mosaic_09cm_area" + str(num) + ".tif"
    top_image = io.imread(filename) 
    return top_image


def get_dsm_image(num):
    filename = os.path.expanduser("~/Pictures/project") + "/ndsm/ndsm_09cm_matching_area" + str(num) + ".bmp"
    dsm_image = io.imread(filename)
    return dsm_image


def get_gt_image(num):
    filename = os.path.expanduser("~/Pictures/project") + "/gts/top_mosaic_09cm_area" + str(num) + ".tif"
    gt_image = io.imread(filename)
    return gt_image


def get_result_image(num):
    filename = os.path.expanduser("~/Pictures/project") + "/classified/result" + str(num) + ".tif"
    result_image = io.imread(filename)
    return result_image


def get_graph(num):
    filename = os.path.expanduser("~/Pictures/project") + "/features/graph" + str(num)    
    graph = cPickle.load(open(filename, "rb"))
    return graph


def get_importance(num):
    filename = os.path.expanduser("~/Pictures/project") + "/features/neighbours_importance" + str(num)    
    importance = cPickle.load(open(filename, "rb"))
    return importance


def get_classes_dist(num):
    filename = os.path.expanduser("~/Pictures/project") + "/features/classes_dist" + str(num)
    dist = cPickle.load(open(filename, "wb+"))
    return dist


def make_classified_image(num, classes_of_segments):
    pathname = os.path.expanduser("~/Pictures/project") + "/classified"
    if not os.path.exists(pathname):
        os.mkdir(pathname)

    list_of_segments = get_list_of_segments(num)
    new_image = get_top_image(num)

    for segment, segment_class in zip(list_of_segments, classes_of_segments):
        for pixel in segment:
            x = pixel[0]
            y = pixel[1]
            new_image[x][y] = np.array(object_types[segment_class])
                
    filename = os.path.expanduser("~/Pictures/project") + "/classified/result" + str(num) + ".tif"
    io.imsave(filename, new_image)


def get_set(data_set):
    x = []
    y = []
    for num in data_set:
        x += get_features_value(num)
        y += get_segments_classes(num)
    x = np.array(x)
    y = np.array(y)
    x = [(np.atleast_2d(vect), np.empty((0, 2), dtype=np.int)) for vect in x]
    y = y.reshape(-1, 1)
    return x, y
