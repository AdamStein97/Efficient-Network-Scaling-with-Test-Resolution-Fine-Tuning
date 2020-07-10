import os
import numpy as np
import tensorflow as tf
from PIL import Image
import classifier as c
from classifier.utils import load_config
from classifier.tf_models.classifier_model import ImageClassifier
from classifier.preprocessor import Preprocessor

config = load_config()

preprocessor = Preprocessor(**config)

model = ImageClassifier(**config)

inp = tf.zeros((64, 64, 64, 3))
_ = model(inp)

model.load_weights(os.path.join(c.MODEL_DIR, "fine_tuned_classifier"))

image_files = [f for f in os.listdir(c.IMG_DIR)]
image_files.sort()

for image_file in image_files:
    pillow_image = Image.open(os.path.join(c.IMG_DIR, image_file))
    im = {"image": np.array(pillow_image), "label": -1}

    input_image, _ = preprocessor.test_preprocess(im)

    label = model(tf.expand_dims(input_image, axis=0), training_mbconv=False, training_classification_layers=False).numpy()[0]

    print("Image: {} -> Cat : {:.4f} %, Dog : {:.4f} %".format(image_file, label[0], label[1]))


