import os
import tensorflow as tf
import classifier as c
from classifier.utils import load_config
from classifier.tf_models.classifier_model import ImageClassifier

config = load_config()

model = ImageClassifier(**config)

inp = tf.zeros((64, 64, 64, 3))
_ = model(inp)

model.load_weights(os.path.join(c.MODEL_DIR, "fine_tuned_classifier"))

