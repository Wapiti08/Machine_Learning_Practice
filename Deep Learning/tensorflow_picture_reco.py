import tensorflow as tf


mnist = tf.keras.datasets.mnist


(x_train, y_train),(x_test, y_test) = mnist.load_data()
# this is the class can stop the epochs when the condition is acceptable
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch,logs={}):
    if (logs.get('acc')>=0.99):
      print("the result has reached the 99% so cancelling the epoch")
      self.model.stop_learning = True
x_train,x_test = x_train/255.0, x_test/255.0
callbacks = myCallback()

model = tf.keras.models.Sequential([
# flatten means to change the shape of picture
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_test,y_test,epochs = 10,callbacks=[callbacks])

