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
# print(a)
# all train images are 28 * 28 and 60000 in number 


b = len(train_labels)
# print(b)
# length of labels are 60000 for 60000 images 


c = train_labels
# print(c)
# labels between 0-9 


d = test_images.shape
# print(d)
# 10000 images in test and 28 *28 


e = len(test_labels)
# print(e)
# 10000 labels for 10000 test images 

# step 3 end 


# step 4 start 

# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(True)
# plt.show()
# this show the given images of dataset with color bar 0 to 255 


# step 4 end 

# step 5 start

# we have to scale thye all images from 0 to 1 from 0 to 255 .
#  to do so we have to divide from 255 
#   and its important that training set and testing set all set in one format  


train_images = train_images / 255.0

test_images = test_images / 255.0

# plt.figure()
# plt.imshow(train_images[2], cmap=plt.cm.binary)
# plt.xlabel(class_names[train_labels[2]])
# # plt.colorbar()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# step 5 end  

# step 6 start 

# to train nueral network we have to create own model on own properties 
# model is combination of own layers properties 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # this used to flattern the layer from 28*28 to 28 * 28 = 784 pixels one dimenstional picture
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)

    # After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons). The second (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.
])

# compile the model 

# Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:

# Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# Optimizer —This is how the model is updated based on the data it sees and its loss function.
# Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# feed the model 
model.fit(train_images, train_labels, epochs=10)


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
# step 6 end 