import tensorflow as tf

#Create a graph
g=tf.Graph()

#Establish the gragh as the "default" graph
with g.as_default():
    #Assemble a graph consisting of the following three operations:
    # *two tf.constant operations to create the operands
    # *One tf.add operation to add the two operands
    x=tf.constant(32,name='x_const')
    y=tf.constant(5,name='y_const')
    sum=tf.add(x,y,name="x_y_sum")

    #Now create a session
    #The seesion will run the default graph
    with tf.Session() as sess:
        #method 1
        print(sum.eval())
        #method 2
        print(sess.run(sum))