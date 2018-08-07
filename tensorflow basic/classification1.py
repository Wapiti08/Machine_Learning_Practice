import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#read 1 to 10 data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    #generate the predicted value,[1,10]
    y_pre=sess.run(prediction,feed_dict={x:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre),tf.argmax(v_ys))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={ys:v_ys,xs:v_xs})
    return result

def add_layer(inputs,in_size,out_size,activation_function=None):
    #定义一个矩阵，in_size表示行，out_size表示列
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    result=tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function is None:
        outputs=result
    else:
        outputs=activation_function(result)
    return outputs

#define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,784]) #28*28
ys=tf.placeholder(tf.float32,[None,10])
#add output layer
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)
#the error between prediction and real data
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#session and initialize
sess=tf.Session()
sess.run(tf.initialize_all_variables)

for i in range(1000):
    #get the data from the mnist
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        #dividing the data into two parts,mnist includes the training 
        #data and testing data
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
