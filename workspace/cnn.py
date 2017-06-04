from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from namespace import *
from keras import losses



def segment_to_feature_vector(image, segment):
    tile = []
    for pixel in segment:
        x = pixel[0]
        y = pixel[1]
        tile.append(image[x][y])
    return tile


def CNN(train_set):
    data = []
    labels = []

    MIN_SEG_LEN = 10000000000;
    for num in train_set:
        image = get_top_image(num)
        list_of_segments = get_list_of_segments(num)
        classes  = get_segments_classes(num)
        for segment, seg_class in zip(list_of_segments, classes):
            features = segment_to_feature_vector(image, segment)
            MIN_SEG_LEN = min(MIN_SEG_LEN, len(features))
            data.append(features)
            labels.append(seg_class)

    for i in range(len(data)):
        data[i] = data[i][: MIN_SEG_LEN]

    data = np.array(data) / 255.0
    
    # transforming the labels into vectors in the range [0, num_classes] -- this
    # generates a vector for each label where the index of the label
    # is set to `1` and all other entries to `0`
    labels = np_utils.to_categorical(labels, CNT_OF_CLASSES)

    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(Dense(768, input_dim=MIN_SEG_LEN, init="uniform", activation="relu"))
    model.add(Dense(384, init="uniform", activation="relu"))
    model.add(Dense(CNT_OF_CLASSES))
    #model.add(Dense(2))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    model.fit(trainX, trainY, nb_epoch=50, batch_size=128, verbose=1)
    cPickle.dump(model, open("cnn", "wb+"))
    return model

def classify(test_set, model):
    for num in test_set:
        print "classification of picture ", num
        X = get_features_value(num)
        X = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
        y = np.vstack(model.predict(X))
        y = y.reshape(len(y))
        y = list(y)
        make_classified_image(num, y)









