import tensorflow as tf
import numpy as np

class OldMaskLayer(tf.keras.layers.Layer):
    """
    Old class for mask layer (not optimized) used to filter out noisy activation function.
        - img_size: size of the input feature maps (n*n)
        - depth:    input's depth (# of feature maps)
        - call():   performs the masking task
        - WARN:     this current version uses ugly for loops
    """
    def __init__(self, img_size=14, depth=512):
        super(OldMaskLayer, self).__init__(trainable=False, dynamic=True)
        self.img_size = img_size
        self.depth = depth
        self.shape = (img_size, img_size, depth)
        self.tau  = 0.5/(img_size*img_size)
        self.beta = 4
        aux = tf.zeros_initializer()
        self.masked_filters = tf.Variable(
            initial_value=aux(shape=(8,14,14,512), dtype='float32'),
            trainable=False)
    '''
    vecchia build della nuova classe
    def build(self, input_shape): 
        aux = tf.zeros_initializer()
        self.masked_filters = tf.Variable(
            initial_value=aux(shape=input_shape[1:], dtype='float32'),
            trainable=False,
            validate_shape=False
        )
        self.mask_tensor = tf.Variable(
            initial_value=aux(shape=input_shape[1:], dtype='float32'), 
            trainable=False
        )
        # to compute masks 
        x = tf.constant([np.arange(0,self.img_size,1) for i in range(self.img_size)])
        y = tf.transpose(x)
        self.col_mat = tf.stack([x]*512, axis=2)
        self.row_mat = tf.stack([y]*512, axis=2)
    '''

    def call(self, inputs):                         # the computation function
        temp = np.zeros(shape=self.shape)
        for b in range(1):
            for z in range(self.depth):
                feature_map = tf.slice(inputs[b],[0,0,z],[self.img_size,self.img_size,1])   # select just one matrix of the 512
                mu = self.__argmax(tf.reshape(feature_map, shape=[-1,1]))           # find max indices in the (flattened) feature map
                mask = self.__compute_mask(mu, self.img_size)                       # compute mask centered in those indeces
                masked_output = tf.math.multiply(feature_map,mask).numpy()          # apply corresponding mask
                
                for i in range(self.img_size):                                      # copy masked feature map in the data structure
                    for j in range(self.img_size):
                        temp[i,j,z] = masked_output[i,j,0]     
            
            self.masked_filters[b].assign(temp)
        return self.masked_filters  # tf.reshape(self.masked_filters, [-1, self.img_size, self.img_size, self.depth])  
    

    
    def compute_output_shape(self, input_shape):    # required!
        return input_shape                          # masking doesn not change the output shape


    def get_config(self):                           # to print new class attribute
        cfg = super().get_config()
        cfg['img_size'] = self.img_size
        cfg['depth']    = self.depth
        cfg['shape']    = self.shape
        cfg['tau']      = self.tau
        cfg['beta']     = self.beta
        return cfg


    def __argmax(self, flatten_feature_map):    
        mu = tf.math.argmax(flatten_feature_map, 0)
        row = mu // self.img_size
        col = mu % self.img_size
        return row, col


    def __compute_mask(self, mu, n, tau=1, beta=1):
        i_max = mu[0]
        j_max = mu[1]
        mat = np.zeros(shape=(n,n,1))
        for i in range(n):
            for j in range(n):
                mat[i,j] = self.tau * max(1-self.beta*(abs(i-i_max)+abs(j-j_max))/n, -1)
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