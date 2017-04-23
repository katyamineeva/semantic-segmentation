<h3 align="center">
Mineeva Ekaterina  <br />
NRU HSE CSD AMI-152 <br />
Project  <br />
</h3>
<h2 align="center">
«Semanic segmentation of images  <br />
in terms of automatic labeling aerial photographs»  <br />
</h2>
<h3 align="center">
Mentor -- Vadim Gorbachev
</h3>
 <br />

#### Task

Analyze labeled aerial photographs and automate classification of objects on the image. 
In this project dataset include high quality aerial photographs of German cities Vaihingen and Potsdam. Each pixel should be classified as a member of the following types of objects:


* Impervious surfaces
* Building
* Low vegetation
* Tree
* Car
* Clutter/background

Each aerial photograph is complemented by: 


* TOP – true orthophoto
* DSM – digital surface model
* nDSM – normalized DSM
* PAN – pan-chromatic image (grayscale)
* CII – color-infrared image 


Moreover, Ground Truth images (manually labeled) are provided for some aerial photographs. It allows to apply methods of Machine Learning to that problem. Also currently, [contest](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html) are opened and everyone can compete with other participants in accuracy of their solutions.

 <br />

#### Methods

For solving this problem supervised Machine Learning was applied. Process of segmentation can be divided into several steps:

1. Presegmentation of image into small "superpixels".
Due to the high resolution of images, analyzing each pixel separately from others requires hard time-consuming computations. Futhermore, features of a group of similar pixels give more information than the features of the only one pixel. As an algorithm for presegmentation was chosen SLIC (Simple Linear Iterative Clustering) due to its efficiency and consistency of segments, it produces.

2. Indicating features.
At this moment was indicated following features:
    * color
    * color of adjacent pixels
    * maximum value for each channel in RGB model
    * height of objects
    * variance of height
    * shape and edges of objects
    * probability of each class 
    * etc.

3. Choice of the classifier.
Currently, Random Forest Classifier was chosen, however, it's planed to replace it with more advanced one in the future.

4. Analysis of accuracy, f1 score, precision and recall and further development and optimisation.

 <br />

#### Implementation steps

1. First checkpoint - 17 December 2016:
    * Specification of the problem and learning about methods of Machine Learning
    * implementation of presentation and evaluation of error

2. Second checkpoint - 25 March 2017:
    * Choice of features 
    * Implementation based on Random Forest classifier
    * Evaluation of results of segmentation
    * Plan of further development

3. Third checkpoint - 29 May - 3 June 2017
    * Implementation of more complicated features and methods 
* Submitting the final version in the contest
