# **Behavioral Cloning**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/CarND_BehavioralCloning_Model.png "Model Visualization"
[image2]: ./examples/image_augmentation.png "Augmentation"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Included Files

I include the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model_weights.h5` containing a trained convolution neural network
* `track1.mp4` video of car autonomously driving track 1
* `track2.mp4` video of car autonomously driving track 2

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model_weights.h5
```

#### File summaries

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The `model_weights.h5` contains the model weights not the full model. (When I tried saving the full model with keras, it gave some strange error.) To create the full model use
```sh
from model import create_model
model = create_model('model_weights.h5')
```

I modified the original `drive.py` file to correctly import my model and to also send the car speed to the model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In designing the model, I first started with a very simple model with maybe three convolution layers followed by a few fully connected. Though this model had a low mean square error (mse), the car oscillated strongly when testing on the first track and it was not able to stay on the road.

I next tried a larger architecture, like that from the Nvidia paper; however, with this model, the car could not successfully stay on the road for both tracks 1 and 2. I tried modifying this model by using a recurrent neural network layer (LSTM) before the final output (and then modified the drive.py file to give multiple images to the model); however, this model also failed. In both of these models, I also used the car speed as in imput to the fully connected (or LSTM) layer before the network output. I included the speed because when testing the models, I noticed that the speed of the car when collecting training data and the speed of the car when testing, which is set in the drive.py file, should be the same. When I collected training with a speed of 30 and then tested at 9, the car would make too large adjustments, and when training at 9 and testing at higher speeds it would adjust too slow.

The next (and last) model architecture I tried used a first convolution layer followed by four inception based layers and finished with three fully connected and one dropout layer. This model has many fewer parameters as the previous models (only ~172K), but successfully stayed on the road for both tracks!

The final model architecture can be seen in the image below.
* Each convolution and dense layer (except for the last) uses an exponential linear unit (ELU) for an activation. The last layer has a linear output.
* The orange boxes on the right give the output size of each layer.
* The strides are all 1 unless noted using (.,.).
* The Normalization step uses a local average of the image brightness to normalize the image intensity and then zero centers the image mean.
* The dropout layer used used to help prevent over fitting.

![alt text][image1]

_Note: I noticed after finishing that my inception layers are not quite the same as in the GoogLeNet paper - the convolution after the MaxPooling should be 1x1, but I used 3x3._

A table showing more details of the architecture can be obtained by running
```sb
from model import create_model
model = create_model()
print(model.summary())
```

#### 3. Creation of the Training Set & Training Process

**Training data:**
As noted in the previous section, I found that the speed of the car when collecting training data and the speed of the car during training should be about the same.; thus, I used the speed as an input to the model.

For this reason, I collected training data at three different speeds: 5, 15, and 30. I did two laps on each track at 30 and 15, and one lap on each track at 5. For all laps I drove as close to the center of the lane as possible. Note that because the frame acquisition is the same regardless of speed, I had the least training data at 30 and the most training data at 5. The total number of frames (centered images) was 35352.

When training the model, I randomly selected one of the three camera views for each image in a batch. I modified the driving angle for the left and right views by +-0.4 degrees. I then applied a random augmentation to the training data. (Examples show below)
* Translation: I translated the image left or right and up or down by a random number about in the range [-20,+20] pixels. When translating left or right, I modified the steering angle by 0.1*tx, where tx is the horizontal offset in pixels.
* Rotation: I rotated the image by a random angle in the range of [-20,+20] degrees, and I modified the steering angle by 0.07*th, where th is the random rotation angle. Rotation is useful because it can create sharper angles than translation.
* Shadows: I added in two random shadows to each image. One random shadow goes from the top to the bottom and covers about half of the image with some angle; the second shadow is a random square patch.
* Flipping: 50% of the time I horizontally flip and image and change the sign of the steering angle.

I also add a small normally distributed random number to the steering angle (sigma=0.01); this will help smooth out small steering angle errors from my driving as well as help generalize the model.

![alt text][image2]

**Training:**
I split the data into 80/20 training/validation sets. I then fit the network with the adam optimizer for 5 epochs. Each epoch went through all the training images using batches of 64 frames, and the data was shuffled between each epoch.

### Result

The car successfully stays on the road for both tracks; however, on track 1, the car oscillates back and forth a bit. I found that if I trained the network for a few extra epochs that the oscillation amplitude decreases, but the car no longer stays on the road in track two. So further modification of either the network, the parameters, or the training data would be necessary to get smooth driving on both tracks.

### Summary
I used an Inception based CNN to train a car to drive _both_ of the Udacity Simulator tracks without any tire leaving the road. I used translation, rotation, flipping, and the addition of fake shadows to augment the data I collected, and I used a local brightness correction for normalizing the images.
