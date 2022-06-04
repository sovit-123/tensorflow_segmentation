import numpy as np

from utils.labels import camvid_label_dict

def grayscale_to_rgb(num_arr, colormap=camvid_label_dict):
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2]+(3,))
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

# Function to decode encoded mask labels.
def onehot_to_rgb(onehot, colormap=camvid_label_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,))
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

# Function to one hot encode RGB mask labels.
def rgb_to_onehot(rgb_arr, color_dict=camvid_label_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.float32 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr