import tensorflow as tf

# with tf.Graph().as_default():
#     #Create an 8x2 matrix (2-D tensor)
#     matrix=tf.constant([[1,2],[3,4],[5,6],[7,8],
#                         [9,10],[11,12],[13,14],[15,16]],dtype=tf.int32)
#     #Reshape the 8x2 matrix into a xX8 matrix
#     reshape_2x8_matrix=tf.reshape(matrix,[2,8])
#     #Reshape the 8x2 matrix into a 4x4 matrix
#     reshape_4x4_matrix=tf.reshape(matrix,[4,4])

#     with tf.Session() as sess:
#         print("Original matrix (8X2)")
#         print(matrix.eval())
#         print("Reshaped matrix (2x8)")
#         print(reshape_2x8_matrix.eval())
#         print("Reshaped matrix (4x4)")
#         print(reshape_4x4_matrix.eval())

a=tf.constant([5,1,2,7,5,4])
b=tf.constant([4,3,1])

a1=tf.reshape(a,[2,3])
b1=tf.reshape(b,[3,1])
result=tf.matmul(a1,b1)
with tf.Session() as sess:
    print(result.eval())
    print(sess.run(result))
