
# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visuals.png "Visualization"
[image2]: ./images/numSample_per_class.png "ClassHistogram"
[image3]: ./images/normalized.png "Normalized"
[image4]: ./images/after_balance.png "Balanced Histogram"
[image5]: ./images/rotated.png "Augmented"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rajatthomas/self-driving/tree/master/P2)

One thing to mention upfront is that although using **Keras** would have made things easier. I wanted to stick close to using pure TF.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. 

Data are summarized using simple python (numpy) commands for the arrays and class/labels stats were
estimated using the 'signnames.csv' file loaded into the Pandas library

###### See Cell -2 for details 
* The size of training set is **34799** 
* The size of test set is **12630**
* The shape of a traffic sign image is **(32,32)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Cell-3 and 4 contain simple visualizations of the data. 
Here is an example of four different classes.
![alt text][image1]

Now I wanted to check the *balance* between the classes. So, I created a histogram to plot the
number of samples in each class. The following is the plot:
![alt text][image2]

We observe a large imbalance in the number of samples per class. 
### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Cell-5 contains all of my preprocessing step. 

Firstly, I normalized  the data as the following
RED = RED/(RED+GREEN+BLUE) * 255 and similarly for GREEN and BLUE. The reasons I chose this normalization
can be found on this [website](http://aishack.in/tutorials/normalized-rgb/)

Here are the same traffic sign images from above after they have been normalized.
![alt_text][image3]

But unfortunately this manner of normalizing did not work. So, I resorted to just scaling everthing
all channels to between 0.1 and 0.9 and the accuracy went up dramatically.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data provided for the Project was split into training/validation/test. 

In Cell-5 of the notebook I chose random directions to rotate the images into. Now for all the classes that had 
much fewer samples we augment the dataset. 

Below is an example of an augmented image:

![alt text][image5]

And here is the histogram of number of samples after augmentation.

![alt text][image4]

Finally I have **69387** training images, **4410** validation and **12630** test images.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Cell-7 contains the architecture. I wanted to do two things. One push the LeNet architecture to see how 
well it can do and two, did not want to Keras

My final model is basically a LeNet model consisting of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28X28X6 	|
| RELU					|												|
| Max pooling 2X2	    | 2x2 stride, valid padding, outputs 14x14x6  	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10X10X16   |
| RELU					|												|
| Max pooling 2X2	    | 2x2 stride, valid padding, outputs 5x5x16  	|
| Flatten               | Flattens conv output to vector of len 400     |
| Fully connected		| Input 400 Output 120        					|
| Fully connected       | Input 120 Output 84                           |
| Fully connected       | Input 84 Output 43                            |
| Softmax				| Outputs probabilities        					|



#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
 

The learning rate = 0.001 was a good compromise between  accuracy and speed. I used the average cross-entory loss as the loss function for optimzation. I used an ADAM optimizer, BATCH_SIZE=64 (I played around with 256 and 128) but it seemed liked 64 was faster and as accurate).
The number of Epochs was set to 25.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 12th and 13th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of **0.940**
* test set accuracy of **0.915**


* Architecture: LeNet
* LeNet has been shown to work well on small images (32 X 32) and has been the goto model for many of the early years of Conv Nets. Also, since I was planning to use a CPU for the first project, I did not want to train a much bigger network.
* Indeed without much tweeking I got to a test accuracy which is reasonable given that I wanted to stick to LeNet

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
[image6]: ./newimages/60.png "60 kmph"
[image7]: ./newimages/80.png "80 kmph"
[image8]: ./newimages/keep_left.png "exclamation.png"
[image9]: ./newimages/priority_road.png "prioirty"
[image10]: ./newimages/road_work.png "road_work"

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

I wanted to try an alternative experiment in which I give two signs that are not part of the training classes. In doing so, I wanted to see
what are the classes it will come up with and try and identify why 
the network would have thought that to be the case.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| 60 km/h      		    | yield									| 
| 80 km/h     			| 50 km/h 									|
| keep left			    | keep left									|
| priority road	      	| priority road 			 				|
| road work 			| road work      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This does not compare favorably to the accuracy on the test set. The model might have overfitted. Also, I guess if I had taken a larger BATCH_SIZE like 1024 and run for about 50 EPOCHS (for which I did not use the resources at this time) I would have got a better result.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


```
TopKV2(values=array([[  5.43115079e-01,   4.56884921e-01,   7.45622408e-09,
          9.27588978e-11,   1.46565518e-12],
       [  1.00000000e+00,   1.99677260e-15,   6.32129704e-17,
          2.24113602e-17,   3.56235671e-18],
       [  9.96269226e-01,   3.73072946e-03,   3.61760344e-10,
          2.15533251e-15,   5.08442909e-16],
       [  1.00000000e+00,   1.33372202e-11,   1.85112599e-17,
          1.00625061e-19,   7.72305321e-20],
       [  9.75094438e-01,   2.44131759e-02,   4.86327452e-04,
          5.77183209e-06,   1.62929894e-07]], dtype=float32), indices=array([[13, 32,  1, 38, 14],
       [ 1, 29, 35,  5,  0],
       [39, 33, 13, 38, 35],
       [12,  1, 35, 15, 38],
       [25,  1,  4, 29, 13]], dtype=int32))
```
Cell-14 contains my prediction. But above I have pasted the results of the test.

For the first image (60 km/h), the model is quite unsure and gave a wrong result

| Probability         	|     Prediction	            | 
|:---------------------:|:-----------------------------:| 
| .54         			| Yield   						| 
| .45     				| End of speed limit 			|
| .000					| 50 km/h						|
| .000	      			| Bumpy Road					|
| .000				    | Slippery Road      		    |


For the second image (80 km/h), the model got it wrong
Although the class 80 km/h was in the top-5, it has very low weight.

Images 3,4 and 5 were predicted with extremely high accuracy.

## Final comment

The network did very well in my opinion, given that I minimally tweeked
the LeNet and the preprocessing of the images were rather straightforward. In the future, I will use better network architectures to play around with the dataset.
