import os

model_name='vendor_classify'
# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 128

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['Accentiv (India) Pvt. Ltd', 'ACCESS MOBILE (INDIA) PRIVATE LIMITED',
            'ALEXIS GLOBAL PVT LTD','VOITTO ADS','WONDER IMAGES PVT LTD']
num_classes = len(classes)

# batch size
batch_size = 1

# validation split
validation_size = .16

# Counter for total number of iterations performed so far.
total_iterations = 0

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

dir = os.path.dirname(os.path.realpath(__file__))

train_path = dir+'/data/train1/'
test_path = dir+'/data/test/'
checkpoint_dir = dir+"/models/"
trained_model = dir+"/trained_models"   #tf serving models
train_batch_size = batch_size



