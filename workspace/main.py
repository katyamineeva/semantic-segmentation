from presegmentation import presegmentation
from features import compute_features
from rf import learn, classify
from accuracy_evaluation import compute_accuracy
import cPickle
from time import time
import traceback

def main(train_set = [0], test_set = []):
    
    all_data = list(set(train_set).union(test_set))
    
    for picture_num in all_data:
        print "started computing", picture_num
        
        try:            
            presegmentation(picture_num, 10000, 5)
            print "presegmentated picture ", picture_num 
        except Exception, e:
            traceback.print_exc(file = open("log_main.txt", 'w'))
        

        
        try:
            compute_features(picture_num)
            print "computed features of picture ", picture_num 
        except Exception, e:
            traceback.print_exc(file = open("log_main.txt", 'w'))
    
         
    print "\n\nstart learning\n"
    learn(train_set)
    print "\n\nfinished learning\n"

    print "\n\nstart classification\n"
    classifier = cPickle.load(open("classifier", "rb"))
    classify(classifier, test_set)
    print "\n\nclassification complete\n"
    
    compute_accuracy(test_set)
    



start = time()
main(train_set = [3, 5, 11, 13, 17, 21, 23], test_set = [1, 7, 15, 32])
print "time of computing : ", time() - start


