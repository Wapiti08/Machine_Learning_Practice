# import the libraray
import tensorflow as tf

# call the dataset
mnist = tf.keras.datasets.fashion_mnist

# download the dataset
(training_images, training_lables), (test_images, test-lables) = mnist.load_data()
# normalizing the data
training_images = training_images/255.0
test_images = test_images/255.0

# plot the picture
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

# build the neural network:
# 1.flatten, dense ,dense


# compile the model and fit the model