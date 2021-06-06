# step 1 start


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)
# print(np.__version__)

# step 1 end

# step 2 start

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(np.array(test_labels))

# step 2 end

# step 3 start 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']\

# label of the dataset 0-9 represent by above class_name 0 = T-shirt/top , 1= Trouser
               
a = train_images.shape
print(a)
# all train images are 28 * 28 and 60000 in number 


b = len(train_labels)
print(b)
# length of labels are 60000 for 60000 images 


c = train_labels
print(c)
# labels between 0-9 


d = test_images.shape
print(d)
# 10000 images in test and 28 *28 


e = len(test_labels)
print(e)
# 10000 labels for 10000 test images 

# step 3 end 
