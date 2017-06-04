import cv2
from skimage.util import img_as_float
from skimage import io
import os
import cPickle
import numpy as np
from namespace import *




#---------------------------------------------------------------------------
# neighbour's features




def get_neighbours_info(num, list_of_dicts):
    graph = get_graph(mask, cnt_of_segments, num)
    importance = get_importance(num)

    for model in models:
        for channel_ind in range(1 if (model == DSM) else 3):
            featurename = "average" + model + str(channel_ind)
            new_featurename = "neighbours" + model + str(channel_ind)
            value = 0
            for i in range(cnt_of_segments):
                for j in range(cnt_of_segments):
                    if (i == j):
                        continue

                    
                    value += importance[i][j] * list_of_dicts[j][featurename]
                list_of_dicts[i][new_featurename] = value
    


#--------------------------------------------------------------------
# additional information about images

def compute_graph(num):
    mask = get_mask(num)
    list_of_segments = get_list_of_segments(num)
    cnt_of_segments  = len(list_of_segments)

    graph = [[0 for i in range(cnt_of_segments)] for j in range(cnt_of_segments)]


    for i in range(len(mask) - 1):
        for j in range(len(mask[i]) - 1):
            central_num = mask[i][j]
            low_num = mask[i - 1][j]
            right_num = mask[i][j - 1]

            graph[central_num][low_num] += 1
            graph[low_num][central_num] += 1
            graph[central_num][right_num] += 1
            graph[right_num][central_num] += 1
    filename = os.path.expanduser("~/Pictures/project") + "/features/graph" + str(num)
    cPickle.dump(graph, open(filename, "wb+"))
    return graph


def compute_neighbours_importance(num):
    print "     getting graph"
    graph = get_graph(num)
    cnt_of_segments = len(graph)
    print "     computing row sum"
    row_sum = [(sum(graph[i])) for i in xrange(cnt_of_segments)]
    # it's not simmetric! importance[i][j] -- how significant connection with j  is for i compared to other 
    print "     computing impirtance"
    importance = [[graph[i][j] * 1.0 / row_sum[i] for i in range(cnt_of_segments)] for j in range(cnt_of_segments)]
    print "     saving"
    filename = os.path.expanduser("~/Pictures/project") + "/features/neighbours_importance" + str(num)
    cPickle.dump(importance, open(filename, "wb+"))
    return importance


def segment_class(segment, gt_image):
    cnt_classes = {}
    for obj in object_types:
        cnt_classes[obj] = 0

    for pixel in segment:
        x = pixel[0]
        y = pixel[1]
        obj = tuple(gt_image[x][y])
        cnt_classes[obj] += 1 * (obj in object_types)

    major_obj = max(cnt_classes.items(), key = lambda pair: pair[1])[0]
    return object_types.index(major_obj)



def compute_distibution_of_classes(num, classes):
    dist = [[0 for i in CNT_OF_CLASSES] for j in CNT_OF_CLASSES]
    graph = get_graph(num)
    cnt_of_segments = len(classes)
    for i in range(cnt_of_segments):
        for j in range(cnt_of_segments):
            type_i = classes[i]
            type_j = classes[j]
            dist[type_j][type_i] += graph[i][j]
            dist[type_i][type_j] += graph[i][j]
    for i in range(len(objectTypes)):
        dist[i][i] = np.log(dist[i][i])

    filename = os.path.expanduser("~/Pictures/project") + "/features/classes_dist" + str(num)
    cPickle.dump(dist, open(filename, "wb+"))





def get_info(num):
    print "computing graph", num
    graph = compute_graph(num)
    graph = get_graph(num)
    print "computing importance", num

    importance = compute_neighbours_importance(num)
    list_of_segments = get_list_of_segments(num)

    filename = os.path.expanduser("~/Pictures/project") + "/gts/top_mosaic_09cm_area" + str(num) + ".tif"
    if os.path.exists(filename):
        gt_image = get_gt_image(num)
        segments_classes = [segment_class(segment, gt_image) for segment in list_of_segments]

        filename = os.path.expanduser("~/Pictures/project") + "/features/segments_classes" + str(num)
        cPickle.dump(segments_classes, open(filename, "wb+"))


#---------------------------------------------------------------------------
# Block of colors and DSM list_of_dicts 
# segment -- list of pixels in the segment
# image -- 2d matrix of 3d vectors

def get_model_features(num, segment, model, features_dict, image):
    for channel_ind in range(1 if (model == DSM) else 3):
        average = 0
        max_color = 0
        average_of_squared = 0
        for pixel in segment:
            x = pixel[0]
            y = pixel[1]
            value = image[x][y] if  (model == DSM) else image[x][y][channel_ind]

            max_color = max(max_color, value)
            average +=  value
            average_of_squared += value ** 2

        average_of_squared /= len(segment)
        average /= len(segment)
        variance = average_of_squared - average ** 2


        featurename = "average" + model + str(channel_ind)
        features_dict[featurename] = average

        featurename = "max" + model + str(channel_ind)
        features_dict[featurename] = max_color
        
        featurename = "variance" + model + str(channel_ind)
        features_dict[featurename] = variance


def get_color_dsm_features(num, list_of_segments):
    features_dict = [{} for i in range(len(list_of_segments))]
    # RGB
    print "RGB"
    imageRGB = get_top_image(num)
    for segment, d in zip(list_of_segments, features_dict):
        get_model_features(num, segment, RGB, d, imageRGB)
    #HSV
    print "HSV"
    imageHSV = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2HSV)
    for segment, d in zip(list_of_segments, features_dict):
        get_model_features(num, segment, HSV, d, imageHSV)

    #CIELab
    print "Lab"
    imageLab = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2Lab)
    for segment, d in zip(list_of_segments, features_dict):
        get_model_features(num, segment, CIELab, d, imageLab)

    #DSM
    print "DSM"
    image = get_dsm_image(num)
    for segment, d in zip(list_of_segments, features_dict):
        get_model_features(num, segment, DSM, d, image)

    return features_dict



# ---------------------------------------------------------------------------
# the only one function, which is used outside this file
# collection list_of_dicts together

# list_of_dicts is 1d array of dict
# list_of_dicts[i] is an dict of list_of_dicts of segment i in list_of_segments, sorted by feature name
# key -- name of feature, string
# value -- feature value

# num is picture's number



def compute_features(num):
    print "compute_features", num
    pathname = os.path.expanduser("~/Pictures/project") + "/features"
    if not os.path.exists(pathname):
        os.mkdir(pathname)

    
    list_of_segments = get_list_of_segments(num)

    print "col feat", num

    list_of_dicts = get_color_dsm_features(num, list_of_segments)
        

    print "get info", num
    get_info(num)

    for segment, d in zip(list_of_segments, list_of_dicts):
        get_neighbours_features(num, segment, d)

    features_value = []

    print "dict to list", num
    for d in list_of_dicts:
        val = [d[key] for key in sorted(d.keys())]
        features_value.append(val)

    print "saving", num
    # dict: keys -- feature names
    filename = os.path.expanduser("~/Pictures/project") + "/features/list_of_dicts" + str(num)
    cPickle.dump(list_of_dicts, open(filename, "wb+"))
    
    # lists of feature values only
    filename = os.path.expanduser("~/Pictures/project") + "/features/features_value" + str(num)
    cPickle.dump(features_value, open(filename, "wb+"))

