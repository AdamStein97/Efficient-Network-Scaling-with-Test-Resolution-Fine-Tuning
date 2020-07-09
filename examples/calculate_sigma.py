import tensorflow as tf
from classifier.preprocessor import Preprocessor

preprocessor = Preprocessor()

train_set, _ = preprocessor.load_dataset()

sigma = tf.constant(0.0)
images_seen = 0.0

for example in train_set:
    image, label = example['image'], example['label']
    _, scale = preprocessor.random_resized_crop(image)
    sigma += scale
    images_seen += 1.0

sigma /= images_seen
print(sigma.numpy())
