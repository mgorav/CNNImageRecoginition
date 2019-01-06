# CNNImageRecoginition
#**Deep Learning CNN for Image Recognition**

Deep Learning CNN model which recognizes images. This projects provides generic framework for building image recognition. The framework als uses image augmentation to achieve better results. This program when run locally will take lots of time. Further image size is intentionally chosen less in favour of fast run. Best results will be achieved when ran on GPU. Further accuracy of 85% on the test data was achieved.


_**Image Recognition CNN Architecture**_

The architecture consist of following steps:
_Step 1_ - Convolution (with 32 feature detector & rectifier 0 relu. Since images are colored input_shape will be used 3D array of 64x64X3 (smaller format for local machine run (IN GPU - 256x256X3 can be used which will give better results)

_Step 2_ - Pooling - reducing the size of feature map. Here Max Pooling will be used. To achieve better accuracy - add a second convolution layer

_Step 3_ - Flattening

_Step 4_ - Full connection - PASS Step 3 output to ANN

**NOTE:** Image augmentation for better results i.e. flip, invert etc etc images provided, hence helps in creating different observations of the images

_**How to use CNN Framework? - putting thins into prospective**_

1. Create a director imagedb
2. With in imagedb create two subfolders:
  2.1) training_set
  2.2) test_set
3. With the training_set and test_set create image classification holder like "Gaurav" etc and put your different images inside it.

