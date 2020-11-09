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
# epochs is used as batch 


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# verbose=0 will show you nothing (silent)

# verbose=1 will show you an animated progress bar like this:

# progres_bar

# verbose=2 will just mention the number of epoch like this:
# epochs 1/10

print('\nTest accuracy:', test_acc)
# step 6 end 


# step 7 start 

# make a prediction 

# With the model trained, you can use it to make predictions about some images. The model's linear outputs, logits. Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

# print(predictions[0])

# it print the 10 value of array with its confidence 
# take out highest confident value 
q = np.argmax(predictions[0])
# print(test_images)


# make a function for plot image of dataset and as well as prediction 
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_prediction(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#   verify prediction on image 
# i = 1
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_prediction(i, predictions[i],  test_labels)
# plt.show()



# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_prediction(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# step 7 end 