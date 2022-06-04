import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils.label_convert import grayscale_to_rgb
from utils.labels import camvid_label_dict

def image_overlay(image, segmented_image, name=None):
    
    alpha = 1.0 # Transparency for the original image.
    beta  = 1.0 # Transparency for the segmentation map.
    gamma = 0.0 # Scalar added to each sum.
    
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def display_image_and_mask(data_list):
    plt.figure(figsize=(15, 5))
    title = ['GT Image', 'GT Mask', 'Overlayed Mask']
    print(np.array(data_list[1].shape))
   
    # Create RGB segmentation map from grayscale segmentation map.
    if len(np.array(data_list[1].shape)) == 2:
        segmented = grayscale_to_rgb(np.array(data_list[1]), camvid_label_dict)
    else:
        segmented = np.array(data_list[1])
    # Create the overlayed image.
    overlayed_image = image_overlay(np.array(data_list[0], dtype=np.float32), 
                                    np.array(segmented, dtype=np.float32),
                                   )
    data_list.append(overlayed_image)
    
    for i in range(len(data_list)):
        plt.subplot(1, len(data_list), i+1)
        plt.title(title[i])
        if title[i] == 'GT Mask':
            plt.imshow(np.array(data_list[i]))
        else:
            plt.imshow(np.array(data_list[i])/255.)
        plt.axis('off')
        
    plt.show()