# **Vehicle Detection Project** 

---

**Vehicle Detection Project**

A Self-driving car needs to detect other vehicles on the road to safely drive. The goal of this project is to detect other vehicles on the road using images from the roof top-mounted camera on self driving car.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/car_hog.png
[image3]: ./examples/non_car_hog.png
[image4]: ./examples/sliding_window_test_1.png
[image5]: ./examples/pipelineTEstImages.png
[image6]: ./examples/heatmap1.png
[image7]: ./examples/label1.png
[image8]: ./examples/final_output.png

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first five code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for a car and a non-car image:


![alt text][image2]

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of color space and HOG parameters like orientation, pixels per cell, HOG channels and the cells per block.

I mixed and randomized the vehicles and non-vehicles data set. Then, I bifurcated the randomized mixed data set with 80% training and 20% testing data. I extracted the HOG features to train the SVC and later evaluate the accuracy performance over the test set. I have used the accuracy as the measure of selecting the HOG parameters. With  
```
colorspace = 'YUV' 
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
```
parameters, I achieved test set accuracy of 96.5%, which is best as compared to my other experiments on combination of HOG parameters value.
PS: I got the idea of these value from the work at https://github.com/jeremy-shannon/CarND-Vehicle-Detection

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using just HOG features. The code for training over the training data set lies in 8th cell of the IPython notebook.
I mixed and randomized the vehicles and non-vehicles data set. Then, I bifurcated the randomized mixed data set with 80% training and 20% testing data. I extracted the HOG features to train the SVC and later evaluate the accuracy performance over the test set

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched for a car in the desired region of the image (sampled space from the image and rescalling to 64x64px) with a desired window size (64x64 px) before classifying with my ```svc``` classifier. 

I used the middle half of the images as search space to search for the cars. I have picked the empirical cordinates pixel values to try various scaling window sizes for the car search.
Here is an example of the sliding window overlapped over first test image.

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales (1.0, 1.5, 2.0, 3.5) using YUV 3-channel HOG features in the feature vector, which provided a decent result (not very good).  Here are some example images:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an image showing the heatmap from a test image 1, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image7]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image8]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I could'nt spare much time to this project. But I genuinely feel there is lot of room for improvement as the video output contains a lot of false car detections. the pipeline is likely to fail is varying color combinations of the video. To make it more robust, I think of following pointers:
1. Adding Color and Gradient features in addition to HOG for training the classifier.
2. Experimenting more with the HOG / Color / Gradient parameters.
3. Augmenting training examples of car and non-car images. Better training, better accuracy.
4. Smoothening out the false detection in the video frames.

