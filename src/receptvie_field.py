import math

def outFromIn(conv, layerIn):
    """
    Computes the receptive field in a matrix after a single convolution 
    operation, with conv = [filter size, stride, padding] parameters
    """
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]
    
    n_out = math.floor((n_in - k + 2*p)/s) + 1
    actualP = (n_out-1)*s - n_in + k 
    pR = math.ceil(actualP/2)
    pL = math.floor(actualP/2)
    
    j_out = j_in * s
    r_out = r_in + (k - 1)*j_in
    start_out = start_in + ((k-1)/2 - pL)*j_in
    return n_out, j_out, r_out, start_out
  

def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))
 

def setup_net_arch():
    """
    Precomputes net receptive fields from each layer in the given convolutional architecture
    """
    IMG_SIZE = 224
    CONV_ARCH = [[3,1,1],[3,1,1],[2,2,0],
                [3,1,1],[3,1,1],[2,2,0],
                [3,1,1],[3,1,1],[3,1,1],[2,2,0],
                [3,1,1],[3,1,1],[3,1,1],[2,2,0],
                [3,1,1],[3,1,1],[3,1,1],[2,2,0]]
    layers_info = []
    currentLayer = [IMG_SIZE, 1, 1, 0.5]      # input layer
    for i in range(len(CONV_ARCH)):
        currentLayer = outFromIn(CONV_ARCH[i], currentLayer)
        layers_info.append(currentLayer)

    return layers_info

def receptive_field(layer_name, f_idx):
    """
    Returns receptive field size and center in the input image, given a feature index (i,j) and layer
        - layer_name: (string) layer from which compute the receptive field
        - f_idx     : index (i,j) of the feature at 'layer_name' layer
    """
    VGG16_LAYERS = ["conv1-1","conv1-2","pool1",
                    "conv2-1","conv2-2","pool2",
                    "conv3-1","conv3-2","conv3-3","pool3",
                    "conv4-1","conv4-2","conv4-3","pool3",
                    "conv5-1","conv5-2","conv5-3","pool5"]

    layers_info = setup_net_arch() 
    layer_idx = VGG16_LAYERS.index(layer_name)
    
    n = layers_info[layer_idx][0]
    j = layers_info[layer_idx][1]
    r = layers_info[layer_idx][2]
    start = layers_info[layer_idx][3]
    assert(f_i < n)
    assert(f_j < n)
    
    print ("receptive field: (%s, %s)" % (r, r))
    print ("center: (%s, %s)" % (start+f_i*j, start+f_j*j))


receptive_field("conv5-3", 12, 3)