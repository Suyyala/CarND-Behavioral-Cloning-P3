**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center-driving.png "Grayscaling"
[image3]: ./examples/recovery2.png "Recovery Image1"
[image4]: ./examples/recovery3.png "Recovery Image2"
[image5]: ./examples/recovery4.png "Recovery Image3"
[image6]: ./examples/test_just_crop.jpg "Normal Image"
[image7]: ./examples/test_just_crop_flip.jpg "Flipped Image"
[image8]: ./examples/model-train1.jpg "Flipped Image"
[image9]: ./examples/conv-cnn-arch.jpg "Flipped Image"


###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network based on Nvidia's DAVE-2.
 https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

My network consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. The input image is cropped and processed before passing to the network.

The model includes RELU activation layer  to introduce nonlinearity in each of the convolution layers, and the data is normalized in the model using a Keras lambda layer (code line 35). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 41). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 103-114). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 112).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, training recovery simulation from  car unwanted position. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was using deep learning methods to predict steering angle based on images from camera. The model is trained manually by using simulator and generating data (images and steering angles), which is then used to train model. Image input data contains lot of spatial information and convolution networks are proven(AlexNet, GoogleNet, LeNet..etc) to yeild great results. Hence, I have started with convolution deeplearning networks. 

My first step was to use a convolution neural network model similar to the Nvidia Dave-2. I have started with this model based on Udacity training inputs and also, based on simplied  deep neural network architecture with proven results. Also,  instead of loading the whole dataset in memory, I have implemented 'data generator' to read off the disk and provide batch samples to the model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
As soon as run the network I have runinto errors with 'out of memory' on my GPU even with 6-7GB of RAM. Then I have realized input data being too large to handle on my GPU. I have preprocessed input images and batched 32 images whcih is acceptable on my GPU. 

Once I have overcome the input feeed data issues, I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the input data to include more data which has helped to generalized the model. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I have added training data to recover from cars of the track positions and also, trained the model on 2nd track to more generatlize learning.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 31-47) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Also, in-order to flgiht off left biased steering angle trained data in the simulator, i have re-corded another lap in the opposite direction of driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering road side or off-track to center so that the vehicle would learn to handle the situtations when going of the track. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images with negative steering angle to increase the data set different angle. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 33,666 number of data points. I then preprocessed this data by cropping the height of each image by half to remove scene data and augmented driving data with flipped image to provide more trainign data. With each batch_set size to 32, I have set to 3 epochs, each model generation took about 30 minutes. I have re-run each such step my many times by tweaking various hyper parameters


I finally randomly shuffled the data set 80% of data for training  and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I have started with  number of epochs to 3 to quickly iterate on model performance as each traning and  test taken 40 minutes on my gpu. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Following are my traning results and car was able to drive autonomous without driving off the track.

![alt text][image8]

Also, created video recording of car's autonomous mode driving.

![alt text][image9]



