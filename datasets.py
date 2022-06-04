from cv2 import IMREAD_UNCHANGED
from tensorflow.keras.utils import Sequence
from utils.label_convert import (
    grayscale_to_rgb, 
    rgb_to_onehot
)

import numpy as np
import albumentations as A
import cv2
import glob as glob

# Class for creating training and validation dataset.
class CustomSeg(Sequence):
    
    def __init__(self, batch_size, image_size, image_paths, mask_paths, num_classes, aug):
        
        self.batch_size  = batch_size
        self.image_size  = image_size
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.num_classes = num_classes
        self.aug         = aug
        
        self.x = np.empty((self.batch_size,) + self.image_size + (3,), dtype="float32")
        self.y = np.empty((self.batch_size,) + self.image_size, dtype="int32")

    def __len__(self):
        
        return len(self.mask_paths) // self.batch_size

    def transforms(self, image, mask):
        
        # Functions defining the augmentations.
        train_augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.6, rotate_limit=0, 
                               shift_limit=0.1, p=0.5, border_mode=0),
                               A.OneOf([A.CLAHE(p=1),
                                        A.RandomGamma(p=1),], p=0.5),
                               A.OneOf([A.Blur(blur_limit=3, p=1),
                                        A.MotionBlur(blur_limit=3, p=1),], p=0.5,),
                               A.OneOf([A.RandomBrightnessContrast(p=1),
                                        A.HueSaturationValue(p=1),], p=0.5,),
        ])
        augment = train_augment(image=image, mask=mask)
        return augment['image'], augment['mask']

    
    def resize(self, image, mask):
        resized_data = A.Resize(
            height=self.image_size[0], width=self.image_size[1],
            interpolation=cv2.INTER_NEAREST,
            always_apply=True, p=1
        )
        resized = resized_data(image=image, mask=mask)
        return resized['image'], resized['mask']
    
    def reset_array(self):
        
        self.x.fill(0.)
        self.y.fill(0.)

    def __getitem__(self, idx):
        
        self.reset_array()
        i = idx * self.batch_size
        batch_image_paths = self.image_paths[i : i + self.batch_size]
        batch_mask_paths = self.mask_paths[i : i + self.batch_size]
        
        for j, (input_image, input_mask) in enumerate(zip(batch_image_paths, batch_mask_paths)):

            img = cv2.imread(input_image, IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     
            msk = cv2.imread(input_mask, IMREAD_UNCHANGED)
            if len(msk.shape) == 2:
                # Convert from grayscale to RGB.
                msk = grayscale_to_rgb(msk)
            else:
                msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)

            img, msk = self.resize(img, msk)

            if self.aug:
                img, msk = self.transforms(img, msk)

            msk = rgb_to_onehot(msk).astype('float32')
            self.x[j] = img.astype('float32')
            self.y[j] = msk.argmax(-1).astype('float32')
            
        return self.x, self.y

# This function returns the training and validation data loaders.
def get_loader(
    train_images_path, train_labels_path, 
    valid_images_path, valid_labels_path,
    batch_size=8,
    image_resize=(224, 224),
    num_classes=3,
    aug=False
):
    # Training image and mask paths.
    train_images = sorted(glob.glob(f"{train_images_path}/*.png"))
    train_masks  = sorted(glob.glob(f"{train_labels_path}/*.png"))
    
    # Validation image and mask paths.
    valid_images = sorted(glob.glob(f"{valid_images_path}/*.png"))
    valid_masks  = sorted(glob.glob(f"{valid_labels_path}/*.png"))

    # Train data loader.
    train_ds = CustomSeg(batch_size=batch_size,
                         image_size=image_resize,
                         image_paths=train_images,
                         mask_paths=train_masks,
                         num_classes=num_classes,
                         aug=aug,
                         )
    # Validation data loader.
    valid_ds = CustomSeg(batch_size=batch_size,
                         image_size=image_resize,
                         image_paths=valid_images,
                         mask_paths=valid_masks,
                         num_classes=num_classes,
                         aug=False,
                         )
    
    return train_ds, valid_ds