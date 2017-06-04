import numpy as np
from namespace import *
from pystruct.models import GraphCRF
from pystruct.learners import FrankWolfeSSVM



def learn(train_set):
    X = []
    y = []
    for num in train_set:
        X += get_features_value(num)
        y += get_segments_classes(num)

    X = np.array(X)
        

    X = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    y = np.vstack(y)

    pbl = GraphCRF(inference_method='unary')
    #svm = NSlackSSVM(pbl, C=100)
    svm = FrankWolfeSSVM(pbl, C=10, max_iter=50)

    svm.fit(X, y)

    cPickle.dump(svm, open("classifier", "wb+"))
    return svm


def classify(test_set, svm):
    for num in test_set:
        print "classification of picture ", num
        X = get_features_value(num)
        X = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
        y = np.vstack(svm.predict(X))
        y = y.reshape(len(y))
        y = list(y)
        make_classified_image(num, y)


