import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import yaml

import classifier as c

def load_config(master_config_name='config.yaml'):
    with open(os.path.join(c.CONFIG_DIR, master_config_name)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model_config_file = config['base_model_config']
    trainer_config_file = config['trainer_config']
    with open(os.path.join(c.CONFIG_DIR, trainer_config_file)) as file:
        trainer_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(os.path.join(c.MODEL_CONFIG_DIR, model_config_file)) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)
    config.update(model_config)
    config.update(trainer_config)
    return config


def calculate_sigma(train_set, preprocessor):
    sigma = tf.constant(0.0)
    images_seen = 0.0

    for example in train_set:
        image, label = example['image'], example['label']
        _, scale = preprocessor.random_resized_crop(image)
        sigma += scale
        images_seen += 1.0

    sigma /= images_seen
    return sigma

def visualize(original, augmented=None):
    if augmented is not None:
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.title('Original image')
        plt.imshow(np.squeeze(original), cmap="gray")

        plt.subplot(1,2,2)
        plt.title('Augmented image')
        plt.imshow(np.squeeze(augmented), cmap="gray")
    else:
        fig= plt.figure(figsize=(8,8))
        plt.imshow(np.squeeze(original), cmap="gray")
        plt.axis('off')
        plt.tight_layout()
        plt.show()