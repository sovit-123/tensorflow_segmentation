from models.fcn32s_mini_vgg import fcn32s_mini_vgg
from datasets import get_loader
from utils.visualizations import (
    display_image_and_mask, image_overlay
)
from utils.label_convert import (
    onehot_to_rgb
)
from utils.labels import camvid_label_dict
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import matplotlib.pyplot as plt

train_ds, valid_ds = get_loader(
    train_images_path='../data/dataset_camvid_segmentation/train',
    train_labels_path='../data/dataset_camvid_segmentation/train_labels',
    valid_images_path='../data/dataset_camvid_segmentation/val',
    valid_labels_path='../data/dataset_camvid_segmentation/val_labels',
    num_classes=32
)

# for i, (images, masks) in enumerate(train_ds):
#     if i == 3:
#         break
#     image, mask = images[0], masks[0]
#     display_image_and_mask([image, mask])

model = fcn32s_mini_vgg(num_classes=32, input_shape=(None, None, 3))
print(model.summary())

model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'],
)  

history = model.fit(
    train_ds,
    epochs=80, 
    verbose=1,
    validation_data=valid_ds,
    workers=4,
    use_multiprocessing=True,
    # callbacks=[model_checkpoint_callback],
)

# Inference
def inference(model):
    
    # _, valid_ds = get_loader()
    
    for i, data in enumerate(valid_ds):
        
        batch_img, batch_mask = data[0], data[1]
        batch_mask = to_categorical(batch_mask, num_classes=32)
        pred_all = (model.predict(batch_img)).astype('float32')
        pred_all = pred_all.argmax(-1)
        pred_all = to_categorical(pred_all, num_classes=32)
        batch_img = (batch_img).astype('uint8')
        break

    for i in range(0, len(batch_img)):
        
        fig = plt.figure(figsize=(20, 8))
        
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(batch_img[i])
        ax1.title.set_text('Actual frame')
        plt.axis('off')
        
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.set_title('Ground truth labels')
        ax2.imshow(onehot_to_rgb(batch_mask[i], camvid_label_dict))
        plt.axis('off')
        
        rgb_mask = onehot_to_rgb(pred_all[i], camvid_label_dict)
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.set_title('Predicted labels')
        ax3.imshow(rgb_mask)
        plt.axis('off')

        overlayed_image = image_overlay(batch_img[i], rgb_mask)
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.set_title('Overlayed image')
        ax4.imshow(overlayed_image)
        plt.axis('off')
        
        plt.show()

inference(model)
