import tensorflow as tf
import numpy as np

#add the layer
#activation_function is null -----liner function
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #normally the value of biases is not zero
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus
    else:
        outputs=activation_function(Wx_plus)
    return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0.0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
#difine two lawyers
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)
#the add in tensorflow named reduce_sum
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
#optimizer can imporve the loss for the next prediction
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%50==0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))


