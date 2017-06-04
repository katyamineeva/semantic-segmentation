import cPickle
from sklearn.ensemble import RandomForestClassifier
from namespace import *
import numpy as np



# train_set -- an array, containing numbers of picture, which are in train set
# it is the first function, which uses outside this file

def learn(train_set):
    clf = RandomForestClassifier(n_estimators = 150)
    clf.n_classes_  = len(object_types)

    # x -- 2d array, x[i] -- list of feature values for segment  i
    # y -- 1d array, y[i] -- true class of segment  i
    x = []
    y = []

    for num in train_set:
        x += get_features_value(num)
        y += get_segments_classes(num)

    clf.fit(x, y)
    cPickle.dump(clf, open("classifier", "wb+"))


# -----------------------------------------------------------------
# clf -- classifier, test_set -- an array, containing numbers of picture, which are in test set

def classify(clf, test_set):
    for num in test_set:
        print "classification of picture ", num
        x = get_features_value(num)
        y = clf.predict(x)
        make_classified_image(num, y)

