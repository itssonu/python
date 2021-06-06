# step 1 start


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
# numpy is an aaray buider for python and known as numerical pyhon 

import matplotlib.pyplot as plt
# matplotlib is used as plotting a graphp with array  

print(tf.__version__)
print(np.__version__)


# step 1 end

# step 2 start

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# load data function download the data set and return four numerical array as defined above train_images,train_labels,test_images,test_labels

# print(np.array(test_labels))

# step 2 end


