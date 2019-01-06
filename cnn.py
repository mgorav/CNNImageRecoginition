

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
sequencialClassifier = Sequential()

# Step 1 - Convolution (with 32 feature detector & rectifier 0 relu. Since images
# are colored input_shape will be used 3D array of 64x64X3 (smaller format for
# local machine run (IN GPU - 256x256X3 can be used which will give better results)
sequencialClassifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation ='relu'))

# Step 2 - Pooling - reducing the size of feature map. Here Max Pooling will be used
sequencialClassifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
sequencialClassifier.add(Convolution2D(32, 3, 3, activation ='relu'))
sequencialClassifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
sequencialClassifier.add(Flatten())

# Step 4 - Full connection
sequencialClassifier.add(Dense(output_dim = 128, activation ='relu'))
sequencialClassifier.add(Dense(output_dim = 1, activation ='sigmoid'))

# Compiling the CNN
sequencialClassifier.compile(optimizer ='adam', loss ='binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
# Image augmentation for better results i.e. flip, invert etc etc images provided, hence
# helps in creating different observations of the images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('imagedb/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('imagedb/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

sequencialClassifier.fit_generator(training_set,
                                   samples_per_epoch = 8000,
                                   nb_epoch = 25,
                                   validation_data = test_set,
                                   nb_val_samples = 2000)