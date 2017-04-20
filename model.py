# Data import
import os
import csv

# Image translation/rotation
import cv2

# Numpy
import numpy as np
from numpy.random import rand
from numpy.random import randn
from numpy.random import randint

# Helpers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Keras
from keras.models import Model
from keras.layers import Cropping2D, Lambda, Flatten, Dense, Dropout, Input, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras import backend as K

# Helper dictionary
view_idx = {'left':0, 0:0, 'center':1, 1:1, 'right':2, 2:2}



# ========================================================================
# Define model architecture



def normalizeInput(x):
    from keras.layers import AveragePooling2D

    # Normalize images using a local average of the brighness (computed as
    # color norm) -> this helps remove the interior of shadows, but not the
    # edge of shadows.
    xnorm = K.sqrt(K.sum(x**2,axis=3,keepdims=True))
    xnorm = AveragePooling2D(pool_size=(15,15),strides=(1,1),border_mode='same')(xnorm)
    x /= (xnorm + 1e-2)

    # Center the data
    x -= K.mean(x, keepdims=True)

    return x

# Define an inception layer
def inceptionModule(x, layer_size=(12,16,8,8), reduction_size=(10,5)):

    # Node 1
    x1 = MaxPooling2D((3,3),strides=(1,1),border_mode='same')(x)
    x1 = Convolution2D(layer_size[0],3,3,subsample=(1,1),activation='elu',border_mode='same')(x1)

    # Node 2
    x2 = Convolution2D(reduction_size[0],1,1,subsample=(1,1),activation='elu')(x)
    x2 = Convolution2D(layer_size[1],3,3,subsample=(1,1),activation='elu',border_mode='same')(x2)

    # Node 3
    x3 = Convolution2D(reduction_size[1],1,1,subsample=(1,1),activation='elu')(x)
    x3 = Convolution2D(layer_size[2],5,5,subsample=(1,1),activation='elu',border_mode='same')(x3)

    # Node 4
    x0 = Convolution2D(layer_size[3],1,1,subsample=(1,1),activation='elu')(x)

    return merge([x0,x1,x2,x3], mode='concat', concat_axis=3)

# Create the full model
def create_model(file_name=None):

    main_input = Input(shape=(160,320,3), name='main_input')
    speed_input = Input(shape=(1,), name='speed_input')

    # Normalize input
    x = Lambda(normalizeInput)(main_input)

    # Crop top and bottom of image
    x = Cropping2D(cropping=((50,20), (0,0)))(x)

    # First convolution layer
    x = Convolution2D(24,5,5,subsample=(2,2),activation='elu')(x) # 43, 158, 24

    # Inception layer 1
    x = inceptionModule(x, (8,16,6,6), (8,3))# 43, 158, 36
    x = MaxPooling2D((3,3),strides=(2,2))(x)# 21, 78, 36

    # Inception layer 2
    x = inceptionModule(x, (12,20,8,8), (12,4))# 21, 78, 48
    x = MaxPooling2D((3,3),strides=(2,2))(x)# 10, 38, 48

    # Inception layer 3
    x = inceptionModule(x, (16,24,12,12), (16,6))# 10, 38, 64
    x = MaxPooling2D((3,3),strides=(2,2))(x)# 4, 18, 64

    # Inception layer 4
    x = inceptionModule(x, (20,32,14,14), (20,7))# 4, 18, 80
    x = AveragePooling2D((4,4),strides=(1,1))(x)# 1, 15, 80

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(100, activation='elu')(x)
    x = Dense(20, activation='elu')(x)

    # Add the speed as an input here
    x = merge([x, speed_input], mode='concat', concat_axis=1)

    main_output = Dense(1, name='main_output')(x)

    model = Model(input=[main_input, speed_input], output=main_output)

    if file_name is not None:
        model.load_weights(file_name)

    return model




# ========================================================================
# Image augmentation and generator




def truncated_normal(n, sigma=1, mu=0):
    x = randn(n)*sigma
    x[x>2*sigma]=2*sigma
    x[x<-2*sigma]=-2*sigma
    return x + mu

def random_shadow(img):

    # The idea for shadow augmentation came from 
    # https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9

    # Add wedge like shadow
    xb = rand(2)*img.shape[1]
    G = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    mask = np.zeros(img.shape[0:2])==0
    mask[((G[1]-img.shape[1])*img.shape[0] - np.diff(xb)*(G[0]-img.shape[0]))>=0] = 1

    if rand(1)>0.5:
        mask = mask==1
    else:
        mask = mask==0

    brightnessFactor = rand(1)
    for i in range(3):
        img[:,:,i][mask] = img[:,:,i][mask]*(brightnessFactor+0.2*rand(1))*0.5

    # Add shadow to random square
    idx0 = np.sort(randint(0,img.shape[0],2))
    idx1 = np.sort(randint(0,img.shape[1],2))

    img[idx0[0]:idx0[1],idx1[0]:idx1[1],:] = img[idx0[0]:idx0[1],idx1[0]:idx1[1],:]*(rand(1) + 0.2*rand(1,1,3))*0.7
    return img

def augment_image(X, y, rot=15., trans=5.):

    # X translation direction
    trans_option = np.random.randint(0,3,1) # (left, center, right)
    trans_offset = truncated_normal(1,3.,(trans_option-1.)*trans) # in the range of [-6-trans,6+trans]

    # Rotation direction
    rot_option = np.random.randint(0,3,1) # (cw, center, ccw)
    rot_offset = truncated_normal(1,3.,(rot_option-1.)*rot)

    # Random x translation
    tx = truncated_normal(1,5.,trans_offset) # [-16-trans,16+trans]
    ty = truncated_normal(1,2.,trans_offset)

    # Translate image
    M = np.float32([[1.,0.,tx],[0.,1.,ty]])
    Xt = cv2.warpAffine(X,M,(X.shape[0:2]),borderMode=cv2.BORDER_REPLICATE)
    Xt = cv2.transpose(Xt)

    # Rotate image
    center = tuple(np.array(X.shape[0:2])/2)
    th = truncated_normal(1,7.,rot_offset)

    rot_mat = cv2.getRotationMatrix2D(center,th,1.)
    Xtr = cv2.warpAffine(Xt,rot_mat,Xt.shape[0:2],borderMode=cv2.BORDER_REPLICATE)
    Xtr = cv2.transpose(Xtr)

    # Add shadow
    Xtr = random_shadow(Xtr)

    # Offset steering angle
    angle_offset = 0.01*tx - 0.07*th + truncated_normal(1,0.01,0)
    angle = y + angle_offset

    return Xtr, angle

def generator(samples, batch_size=32, use_all_views=True):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            speeds = []

            for batch_sample in batch_samples:

                if use_all_views:
                    # Randomly select left, center, right view
                    view_option = int(np.random.randint(0,3,1))
                else:
                    view_option = 1

                # Load image
                X,y,s = load_data_entry(batch_sample, view_option, angle_offset=0.4)
                # Augment image
                X,y = augment_image(X, y, rot=5., trans=5.)
                # Add to stack
                images.append(X)
                angles.append(y)
                speeds.append(s)

            # Convert to numpy arrays
            X = np.array(images)
            y = np.array(angles)
            speeds = np.array(speeds)/31. # normalize speed

            # Randomly flip each image
            # (This code can be vectorized, which is why it is not in
            # augment_image)
            flip_selector = rand(X.shape[0]) > 0.5
            X[flip_selector,:] = np.flip(X[flip_selector,:],axis=2)
            y[flip_selector,0] = (-1) * y[flip_selector,0]

            # shuffle data
            X, y, speeds = shuffle(X, y, speeds)

            # output dictionary for model's dual imput
            yield ({'main_input': X, 'speed_input': speeds}, y)




# =======================================================================
# Data import



def load_sample_data(file_names):
    samples = []
    for file_name in file_names:
        with open(file_name) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)

    return samples

def load_data_entry(sample, view='center', angle_offset=0.2):
    y = float(sample[3])
    s = float(sample[6])
    th = float(sample[4])-float(sample[5])
    if view_idx[view]==0.:
        X = cv2.imread(sample[1])
        y = y + angle_offset
    elif view_idx[view]==2.:
        X = cv2.imread(sample[2])
        y = y - angle_offset
    else:
        X = cv2.imread(sample[0])

    return X,y,s


# =======================================================================
# Load, train, save


if __name__ == "__main__":

    # Folders with data
    file_names = ['K:/Udacity_CarND/Behavioral_Cloning_TrainingData/Training_Data_T{}_{}/driving_log.csv'.format(j,i) for j in (2,1) for i in (5,15,30)]

    # Load data
    samples = load_sample_data(file_names)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # Initialize generators
    train_generator = generator(train_samples, batch_size=64)
    validation_generator = generator(validation_samples, batch_size=32)

    # Create model
    model = create_model()
    print(model.summary())

    # Compile and train model
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                validation_data=validation_generator,
                nb_val_samples=len(validation_samples), nb_epoch=5)

    # Save weights
    model.save_weights('model_weights.h5')

    # There is some error if I try to directly save the model using
