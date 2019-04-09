# [Contest](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html) on semanic segmentation of aerial images

Second year course project at [CS department](https://cs.hse.ru/en/) of the Higher school of Ecominics. 

### Goal

Label each pixel of an image with one of the following marks:

* Impervious surfaces
* Building
* Low vegetation
* Tree
* Car
* Clutter/background

### Dataset

Dataset provided by the organizers of the competition include:

* TOP – true orthophoto
* DSM – digital surface model
* nDSM – normalized DSM
* PAN – pan-chromatic image (grayscale)
* CII – color-infrared image 


### Algorithm steps


#### 1. Presegmentation of image into so-called superpixels.

Pre-segmentation allows to extract more imformation from pixels by gathering similar ones together. This procedure also decreases the number of objects and, consequently, improvers classification efficeincy. For pre-segmentation I use SLIC (Simple Linear Iterative Clustering).

#### 2. Features extraction.

In the final version each superpixel is classified based on the following features:
    
 * average color
 * average color of adjacent superpixels
 * maximum value for each channel in RGB model
 * height of objects
 * variance of height

Also, I tried to train a simple CNN to form an embedding of superpixels into vector space. However, this step didn't improve algorithm's performance, therefore I removed it from the final version. (Note, that it was my first experience with CNNs, so maybe, CNN features can improve algorithm's performance and I just did something wrong :)

#### 3. Classification.

In the final version I use Random Forest as a classifier. I aslo tried Gradient Boosting, which gave almost the same results.

### Results

Final accuracy achived by cross-validation is 89%.
