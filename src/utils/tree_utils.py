import numpy as np
import tensorflow as tf

def vectorify(x, g, s):
    """
    Takes two tensors and returns two vectors
        - xx[d] = sum(x[h,w,d])/s_d
        - gg[d] = s_d/L^2 * sum(dy/dx[h,w,d])
        - s_d = E_i[ E_(h,w)[ x[h,w,d] ] ]
    """
    raise NotImplementedError

def vectorify_on_depth(x):
    """
    xx = tf.ones(shape=(2,2,5))
    x = tf.reduce_sum(xx, axis=[0,1])
    # sum over h and w --> outputs a vector of lenght d=5
    """
    return tf.reduce_sum(x, axis=[0, 1])



''' TEST vectorify on depth 
x = tf.random.uniform(shape=[2, 2, 3], minval=1,
                      maxval=5, dtype=tf.int32, seed=42)
print(x)
print(vectorify_on_depth(x))
'''