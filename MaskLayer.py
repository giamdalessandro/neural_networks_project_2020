import tensorflow as tf
import numpy as np

class MaskLayer(tf.keras.layers.Layer):
    """
    Class for mask layer, used to filter out noisy activation function.
        - img_size: size of the input feature maps (n*n)
        - depth:    input's depth (# of feature maps)
        - call():   performs the masking task
        - WARN:     this current version uses ugly for loops
    """
    def __init__(self, img_size=14, depth=512):
        super(MaskLayer, self).__init__(trainable=False, dynamic=True)
        self.n = img_size
        self.depth = depth
        self.shape = (img_size, img_size, depth)
        self.tau  = 1                     # da verificare
        self.beta = 1                     # da verificare

    def build(self, input_shape=(None,14,14,512)):
        b_init = tf.zeros_initializer()
        self.masked_filters = tf.Variable(
            initial_value=b_init(shape=input_shape, dtype='float32'),
            trainable=False)

    def compute_output_shape(self, input_shape):
        ...

    def call(self, inputs):                         # the computation function
        aux = np.zeros(shape=self.shape)
        for z in range(self.depth):
            feature_map = tf.slice(inputs[0], [0,0,z], [self.n, self.n, 1])        # select just one matrix of the 512 in inputs  
            mu = self.__argmax(tf.reshape(feature_map, shape=[-1,1]))           # find max indices in the (flattened) feature map
            mask = self.__compute_mask(mu, self.n)                              # compute mask centered in those indeces
            masked_output = tf.math.multiply(feature_map,mask).numpy()          # apply corresponding mask
            
            for i in range(self.n):                                             # copy masked feature map in the data structure
                for j in range(self.n):
                    aux[i,j,z] = masked_output[i,j,0]     
        
        self.masked_filters.assign(aux) 
        return self.masked_filters      


    def __argmax(self, flatten_feature_map):    
        mu = tf.math.argmax(flatten_feature_map, 0)
        col = mu // self.n
        row = mu % self.n
        return row, col


    def __compute_mask(self, mu, n, tau=1, beta=1):
        i_max = mu[0]
        j_max = mu[1]
        mat = np.zeros(shape=(n,n,1))
        for i in range(n):
            for j in range(n):
                mat[i,j] = self.tau * max(-1, 1-self.beta*(abs(i-i_max)+abs(j-j_max))/n)
        return mat


'''
### Test per la classe MaskLayer

import tensorflow as tf
import numpy as np
n = 5
depth = 3
maskedLayer = MaskLayer(n,depth)
r = tf.random.uniform(shape=(n,n,depth), seed=42, maxval=5, dtype='int32')
l = maskedLayer.call(r)
print(r)
print(l)

'''