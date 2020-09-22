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
        super(MaskLayer, self).__init__(trainable=False, dynamic=True)
        self.img_size = img_size
        self.depth = depth
        self.shape = (img_size, img_size, depth)
        self.tau  = 0.5/(img_size*img_size)
        self.beta = 4


    def build(self, input_shape):
        # to compute masks
        x = tf.constant([np.arange(0,self.img_size,1) for i in range(self.img_size)])
        y = tf.transpose(x)
        self.col_mat = tf.stack([x]*512, axis=2)
        self.row_mat = tf.stack([y]*512, axis=2)

    
    def call(self, inputs):                         # the computation function
        """
        Creates a mask tensor and applies it to the output of the convolutional layer
        """
        batch_size = tf.shape(inputs)[0]
        output = np.zeros([batch_size,14,14,512])
        for b in range(batch_size):
            # finds the row and col indices of the maximum value across the depth of the tensor
            rows_idx = tf.math.argmax(tf.reduce_max(inputs[b], axis=1), output_type=tf.int32)
            cols_idx = tf.math.argmax(tf.reduce_max(inputs[b], axis=0), output_type=tf.int32)

            # scales up the previous tensors from (1,1,512) to (14,14,512) mantaining the same values
            rows_idx_tensor = tf.tile([[rows_idx]], [14,14,1])
            cols_idx_tensor = tf.tile([[cols_idx]], [14,14,1])

            # performs the absolute value between the tensor just created and another one with
            # the values equal to the row (or col) index (self.row_mat and self.col_mat)
            # these two are "dummy" tensor that serves only to perform this computation easily
            abs_row = tf.math.abs(tf.math.subtract(self.row_mat,rows_idx_tensor))
            abs_col = tf.math.abs(tf.math.subtract(self.col_mat,cols_idx_tensor))

            # val = 1 - B*(abs_row + abs_col)/n
            val = tf.math.subtract(1, (self.beta/self.img_size) * tf.dtypes.cast(tf.math.add(abs_row,abs_col),tf.float32))
            
            # output[b] = max( [T * max(val, -1)] * conv_output[b], 0)
            output[b]=tf.math.maximum((tf.math.multiply(self.tau * tf.math.maximum(val,-1), inputs[b])), 0)
        return tf.convert_to_tensor(output)

    
    def compute_output_shape(self, input_shape):    # required!
        return input_shape                          # masking doesn not change the output shape


    def get_config(self):                           # required!
        cfg = super(MaskLayer, self).get_config()
        cfg.update({
            "img_size": self.img_size,
            "depth"   : self.depth,
            "shape"   : self.shape,
            "tau"     : self.tau,
            "beta"    : self.beta
        })
        return cfg


    @classmethod
    def from_config(cls, config):
        return cls(**config)
