import numpy as np
import time

'''
This is the comparation of time complexity between vector and scalar
'''
#1.product of vector
a=np.random.rand(1000000)
b=np.random.rand(1000000)
tic=time.time()
c=np.dot(a,b)
toc=time.time()
print(c)
print("vectorization:"+str(1000*(toc-tic))+"ms")
print("vectorization:"+str((toc-tic))+"ms")

#2.product of scalar
c=0
tic1=time.time()
for i in range(1000000):
    c+=a[i]*b[i]
toc1=time.time()
print(c)
print("scalar:"+str(1000*(toc1-tic1))+"ms")

'''
the process to normalize the matrix
'''
def normalizeRows(x):    
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    # ord=2 means 2-norm(largest sing.value) 1 mean the max(sum(abs(x),axis=0))
    # ord=2: sqrt(x1*x1+x2*x2+...+xn*xn) 
    # ord=1: abs|x1|+...+abs|x2|
    # axis=0 column 1 row
    # keepdims ---- keep the two-dimensional characteristics of the matrix
    x_norm = None
    x_norm=np.linalg.norm(x,ord=2,axis=1,keepdims=True)
    # Divide x by its norm.
    #x = None
    x=x/x_norm
    ### END CODE HERE ###

    return x






















