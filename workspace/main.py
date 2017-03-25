from __future__ import division
from presegmentation import presegmentation, get_list_of_segments, slic_segmentation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from accuracy_evaluation import accuracy
from skimage import io
import numpy as np
import cv2



IMP_SURF = (255, 255, 255)
BUILDING = (0, 0, 255)
LOW_VEG = (0, 255, 255)
TREE = (0, 255, 0)
CAR = (255, 255, 0)
CLUTTER = (255, 0, 0)

objectTypes = [IMP_SURF, BUILDING, LOW_VEG, TREE, CAR, CLUTTER]


def get_graph(mask, cnt_of_segments, num):
    graph = np.zeros((cnt_of_segments, cnt_of_segments))
    for i in range(len(mask) - 1):
        for j in range(len(mask[i]) - 1):
            central_num = mask[i][j]
            low_num = mask[i - 1][j]
            right_num = mask[i][j - 1]

            graph[low_num][central_num] += 1
            if central_num != low_num:
                graph[central_num][low_num] += 1

            graph[right_num][central_num] += 1
            if central_num != right_num:
                graph[central_num][right_num] += 1
                
    graph.tofile("graph_" + str(num) + ".txt", ", ")
    return graph

def compute_distibution_of_classes(numbers_of_images, list_of_segments):
    dist = np.zeros((len(objectTypes), len(objectTypes)))
    for num in numbers_of_images:
        graph = np.fromfile("graph_" + str(num) + ".txt")
        for i in range(len(graph)):
            for j in range(len(graph[i])):
                gt_image = cv2.imread("gt" + str(num) + ".tif")
                iclass = segment_class(list_of_segments[i], gt_image)
                jclass = segment_class(list_of_segments[j], gt_image)

                dist[iclass][jclass] += graph[i][j]
    for i in range(len(objectTypes)):
        dist[i][i] = np.log(dist[i][i])
        

def potts_potentials():
    pass


def get_neighbours_info(mask, cnt_of_segments, color0, color1, color2, num):
    graph = get_graph(mask, cnt_of_segments, num)
    neighbours_color0 = [0 for k in range(cnt_of_segments)]
    neighbours_color1 = [0 for k in range(cnt_of_segments)]
    neighbours_color2 = [0 for k in range(cnt_of_segments)]
    for i in range(cnt_of_segments):
        for j in range(cnt_of_segments):
            if (i == j):
                continue

            importance = graph[i][j] / sum([graph[i][t] for t in range(cnt_of_segments)])

            neighbours_color0[i] += (color0[j] - 255 * 0.5) * importance + 255 * 0.5
            neighbours_color1[i] += (color1[j] - 255 * 0.5) * importance + 255 * 0.5
            neighbours_color2[i] += (color2[j] - 255 * 0.5) * importance + 255 * 0.5

    return [neighbours_color0, neighbours_color1, neighbours_color2]


def get_color(segment, num, color_ind):
    top_image = cv2.imread("top" + str(num) + ".tif")
    color = 0
    for i in range(len(segment)):
        pixel = segment[i]
        x = pixel[0]
        y = pixel[1]
        color += top_image[x][y][color_ind]
    color = color / len(segment)
    return color


def DSM(segment, num):
    average = 0
    dsm = cv2.imread("dsm" + str(num) + ".tif", cv2.CV_16UC1)
    for pixel in segment:
        x = pixel[0]
        y = pixel[1]
        average += dsm[x][y]
    return average / len(segment)



def get_features(list_of_segments, mask, cnt_of_segments, num):
    color0 = []
    color1 = []
    color2 = []

    for i in range(cnt_of_segments):
        color0.append(get_color(list_of_segments[i], num, 0))
        color1.append(get_color(list_of_segments[i], num, 1))
        color2.append(get_color(list_of_segments[i], num, 2))

    features = [color0, color1, color2]
    features += get_neighbours_info(mask, cnt_of_segments, color0, color1, color2, num)

    average_dsm = [DSM(segment, num) for segment in list_of_segments]
    features += average_dsm

    features = np.asarray(features)

    features.tofile("features_" + str(num) + ".txt", ", " )

    #return [color0, color1, color2]
    return features




def segment_class(segment, gt_image):
    cntOfClasses = dict.fromkeys(objectTypes, 0)
    for i in range(len(segment)):
        pixel = segment[i]
        x = pixel[0]
        y = pixel[1]
        typeOfPixel = tuple(gt_image[x][y])
        cntOfClasses[typeOfPixel] += 1

    major = 0
    for i in range(len(objectTypes)):
        if cntOfClasses[objectTypes[i]] > cntOfClasses[objectTypes[major]]:
            major = i
    return major




def learn(numbers_of_images):
    clf = RandomForestClassifier(n_estimators=15)
    clf.n_classes_ = len(objectTypes)
    for num in numbers_of_images:
        photo_gt = "gt" + str(num) + ".tif"
        gt_image = cv2.imread(photo_gt)
        mask = presegmentation(num)
        #mask = np.fromfile("mask_of_" + str(num) + "_image.txt")
        #mask = np.ndarray(buffer=mask, shape=gt_image.shape)

        cnt_of_segments = max([max(row) for row in mask]) + 1

        #list_of_segments = np.fromfile("list_of_segments_" + str(num) + ".txt")
        #list_of_segments = np.asarray(list_of_segments, shape=(cnt_of_segments,  , 2))
        list_of_segments = get_list_of_segments(mask, cnt_of_segments, num)
        
        features = get_features(list_of_segments, mask, len(list_of_segments), num)
        cnt_of_features = len(features)
        x = [[features[i][j] for i in range(cnt_of_features)] for j in range(cnt_of_segments)]

        y = []
        for i in range(len(list_of_segments)):
            y.append(segment_class(list_of_segments[i], gt_image))

        clf.fit(x, y)
        
    scores = cross_val_score(clf, x, y)
    print "score on training samples", scores.mean()
    return clf


def make_classified_image(list_of_segments, classes, dimensions):
    new_image = np.ndarray(dimensions)
    for i in range(len(list_of_segments)):
        for pixel in list_of_segments[i]:
            x = pixel[0]
            y = pixel[1]
            new_image[x][y] = objectTypes[classes[i]]
    return new_image



def classify(num, clf):
    photo_top =  "top" + str(num) + ".tif"
    top_image = cv2.imread(photo_top)       
    mask = slic_segmentation(filename = photo_top, numSegments = 300)
    #mask = np.fromfile("mask_of_" + str(num) + "_image.txt")
    #mask = np.ndarray(buffer=mask, shape=top_image.shape)
    cnt_of_segments = max([max(row) for row in mask]) + 1
    ##list_of_segments = np.fromfile("list_of_segments_" + str(num) + ".txt")
    list_of_segments = get_list_of_segments(mask, cnt_of_segments, num)

    features = get_features(list_of_segments, mask, cnt_of_segments, num)
    cnt_of_features = len(features)

    x = [[features[i][j] for i in range(cnt_of_features)] for j in range(cnt_of_segments)]
    y = clf.predict(x)

    return make_classified_image(list_of_segments, y, top_image.shape)


def main():
    # clf = learn([...])
    new_image = classify(..., clf)
    accuracy(new_image, cv2.imread(...))

main()
