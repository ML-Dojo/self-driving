


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./edges_in_poly.jpg

---

### Reflection

#### 1. Pipeline


1. Applied the canny algorithm on grayscaled image and then masked it with a polygon that takes the frontol view

2. Hough Transform were used to identify the line. For each line I calculated the slope and used a threshold on the slope to decide if it belongs to left or right lane (more negative implies left and positive the right lanes)

3. lines selected on each right/left lane where then used to select the two extreme end-points through which a line was drawn: (i) list all the lines belonging to a particular lane (left or right), (ii) each line has two (x,y) points (xi1,yi1) and (xi2,yi2) for line 'i' say. (iii) select a (xk,yk) and (xl,yl) such that all other points (x,y) lies (approximately) between them.


![alt text][image1]


#### 2. Potential shortcomings with your current pipeline

1. The threshold chosen seems to work well but it is chosen arbitrarily. It would be nice to have a more principled threshold for that.
2. There is flickering in the line between frames
3. The algorithms needs well annoted roads to work. What if the road markings are missing? 
4. Dangerous scenario: What is the opposing traffic (on a two-way road) comes into your lane for overtaking?

#### 3. Possible improvements to your pipeline

1. It would be nice to use information from previous frames to determine the lane in the current frame
2. Use another metric to calculate mean slope of left/right lanes and draw a parametric line
3. Use information from previous frames to build the lanes. Using a Bayesian framework to select between different possible annotations.
4. Futuristic: If there are other cars in the lane and a communication protocol agreed upon, then the information from the network of cars can be pooled to better the estimation.


```python

```
