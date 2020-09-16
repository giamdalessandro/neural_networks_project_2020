import tensorflow as tf
import numpy as np

class MaskLayer(tf.keras.layers.Layer):
    """
    Class for mask layer, used to filter out noisy activation function.
        - img_size: size of the input feature maps (n*n)
        - depth:    input's depth (# of feature maps)
        - visualize:    if you want fiorellini, put True
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


    def build(self, input_shape): 
        '''TODO'''
        aux = tf.zeros_initializer()
        self.masked_filters = tf.Variable(
            initial_value=tf.zeros_initializer(),
            trainable=False,
            validate_shape=False
        )
        self.mask_tensor = tf.Variable(
            initial_value=aux(shape=[-1] + list(input_shape[1:]), dtype='float32'), 
            trainable=False
        )
        # to compute masks 
        x = tf.convert_to_tensor([np.arange(0,self.img_size,1) for i in range(self.img_size)])
        y = tf.transpose(x)
        self.col_mat = tf.stack([x]*512, axis=2)
        self.row_mat = tf.stack([y]*512, axis=2)

    
    def call(self, inputs):                         # the computation function
        rows_idx = tf.math.argmax(tf.reduce_max(inputs[0], axis=1))
        cols_idx = tf.math.argmax(tf.reduce_max(inputs[0], axis=0))

        self.mask_tensor.assign(self.__compute_mask(rows_idx,cols_idx))

        return tf.math.multiply(self.mask_tensor,inputs[0])

    
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


    def __compute_mask(self, row_idx, col_idx):
        row_idx_tensor = tf.tile(row_idx, [14,14,1])
        col_idx_tensor = tf.tile(col_idx, [14,14,1])

        abs_row = tf.math.abs(tf.math.subtract(self.row_mat,row_idx_tensor))
        abs_col = tf.math.abs(tf.math.subtract(self.col_mat,col_idx_tensor))

        val = tf.math.subtract(1, (self.beta/self.img_size) * tf.math.add(abs_row+abs_col))
        return self.tau * tf.math.maximum(val,-1)