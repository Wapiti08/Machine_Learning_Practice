import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST',one_hot=False)

#Visualize decoder setting
#Parameters
#learing_rate is used to optimizer
learning_rate=0.01
#training times
training_epochs=5
#limit the size of batch
batch_size=256
#used to print the data
display_step=1
#used to show the data in picture
example_to_show=10

#Network Parameters
n_input=784 #MNIST data input(img shape:28*28)

#tf Graph input(only picetures)
X=tf.placeholder("float",[None,n_input])

#hidden layer settings
n_hidden_1=256 #1st layer num features
n_hidden_2=128 #2nd layer num features
weights={
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),

    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input])),
}
biases={
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),

    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2':tf.Variable(tf.random_normal([n_input]))

}

#Building the encoder
def encoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),
    biases['encoder_b1']))
    #Encoder Hidden layer with sigmoid activation #2
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    return layer_2

#Building the decoder
def decoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),
    biases['decoder_b1']))
    #Decoder Hidden layer with sigmoid activation #2
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))
    return layer_2

#Construct model,the result
encoder_op=encoder(X)
decoder_op=decoder(encoder_op)

#Prediction,the result of calculation
y_pred=decoder_op

#Targets(Labels) are the input data,the original data
y_true=X

#Define loss and optimizer,minimize the squared error
cost=tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Initializing the variables
init=tf.global_variables_initializer()

#Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch=int(mnist.train.num_examples/batch_size)
    #Training cycle, epochs is similar to iters
    #there are two cycles,one is every batch,other is the times for practice
    for epoch in range(training_epochs):
        #Loop over all batches
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size) 
            #Run optimization op(backprop) and cost op(to get loss value)
            _,c=sess.run([optimizer,cost],feed_dict={X:batch_xs})
        #Display logs per epoch step,when the epoch<1，exit 
        if epoch%display_step==0:
            print("Epoch:",'%04d'%(epoch+1),
                "cost=","{:.9f}".format(c))
    
    print("Optimization Finished!")

    #Applying encode and decode over test set   [,10]
    encode_decode=sess.run(y_pred,feed_dict={X:mnist.test.images[:example_to_show]})
    #Compare original images with their reconstructions [2,10]
    f,a=mp.subplots(2,10,figsize=(10,2))
    for i in range(example_to_show):
        #[28,28],the true data
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        #the prediction data
        a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
    mp.show()





