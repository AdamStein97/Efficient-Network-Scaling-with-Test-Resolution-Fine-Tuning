import tensorflow as tf
import tensorflow_datasets as tfds


class Preprocessor():
    def __init__(self, base_k_train=64, base_k_image_test=72, base_k_test=64, res_scale=1.0, phi_scaling_factor=1, sigma=0.444, **kwargs):
        res_scale = res_scale ** phi_scaling_factor
        self.k_train = int(round(base_k_train * res_scale))

        self.k_image_test = int(round(base_k_image_test * res_scale))
        self.k_test = int(round(base_k_test * res_scale))

        self.scale_ratio = tf.constant(1.0 / (sigma * (self.k_image_test / self.k_test)),  dtype=tf.float32)

    @tf.function
    def augment(self, image, max_brightness_delta=0.25):
        image = tf.image.random_crop(image, size=[self.k_train, self.k_train, 3])
        image = tf.image.random_brightness(image, max_delta=max_brightness_delta)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.adjust_saturation(image, 3)
        return image

    @tf.function(experimental_relax_shapes=True)
    def scale_image(self, image, min_dim):
        image_shape = tf.shape(image)
        if tf.argmin(image_shape[-3:-1]) == 0:
            scale = image_shape[-3] / min_dim
            scaled_width = tf.cast(tf.cast(image_shape[-2], tf.float64) / scale, tf.int32)
            image = tf.image.resize(image, size=[min_dim, scaled_width], preserve_aspect_ratio=False)
        else:
            scale = image_shape[-2] / min_dim
            scaled_height = tf.cast(tf.cast(image_shape[-3], tf.float64) / scale, tf.int32)
            image = tf.image.resize(image, size=[scaled_height, min_dim], preserve_aspect_ratio=False)
        return image

    @tf.function(experimental_relax_shapes=True)
    def random_resized_crop(self, image,scale_lower=0.2, scale_upper=0.7):
        image_shape = tf.shape(image)
        pixels = tf.cast(image_shape[-3] * image_shape[-2], tf.float32)
        max_scale = tf.cast(tf.minimum(image_shape[-3], image_shape[-2]), tf.float32) ** 2 / pixels
        scale = tf.random.uniform([], minval=tf.minimum(scale_lower, max_scale),
                                  maxval=tf.minimum(scale_upper, max_scale))
        crop_pixels = scale * pixels
        crop_dim = tf.round(tf.math.sqrt(crop_pixels))
        image = tf.image.random_crop(image, size=[crop_dim, crop_dim, 3])
        image = tf.image.resize(image, size=[self.k_train, self.k_train], preserve_aspect_ratio=True)
        return image, scale

    @tf.function(experimental_relax_shapes=True)
    def center_crop(self, image, k_image, k, scale):
        dim = tf.cast(tf.round(tf.cast(k_image, tf.float32) * scale), tf.int32)

        image = self.scale_image(image, dim)
        image = tf.image.resize_with_crop_or_pad(image, dim, dim)
        image = tf.image.central_crop(image, k / k_image)
        return image

    @tf.function(experimental_relax_shapes=True)
    def train_preprocess(self, example, augment_img=True, **kwargs):
        image, label = example['image'], example['label']
        image = tf.image.convert_image_dtype(image, tf.float32)
        image, _ = self.random_resized_crop(image, **kwargs)
        if augment_img:
            image = self.augment(image)
        image = tf.clip_by_value(image, 0, 1)
        return image, label


    @tf.function(experimental_relax_shapes=True)
    def test_preprocess(self, example):
        image, label = example['image'], example['label']
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = self.center_crop(image, self.k_image_test, self.k_test, self.scale_ratio)
        return image, label

    def make_train_datasets(self, train_set, test_set, batch_size=64, **kwargs):
        preprocess_train = lambda x: self.train_preprocess(x, **kwargs)
        preprocess_test = lambda x: self.test_preprocess(x, tf.constant(1.0, dtype=tf.float32), **kwargs)

        train_ds = (train_set
                    .shuffle(4096)
                    .map(preprocess_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    .batch(batch_size, drop_remainder=True)
                    .prefetch(1)
                    )

        test_ds = (test_set
                  .shuffle(4096)
                  .map(preprocess_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                  .batch(batch_size, drop_remainder=True)
                  .prefetch(1)
                  )

        return train_ds, test_ds

    def make_finetune_datasets(self, train_set, test_set, batch_size=64, **kwargs):
        fine_tune_ds = (train_set
                       .shuffle(4096)
                       .map(self.test_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                       .batch(batch_size, drop_remainder=True)
                       .prefetch(1)
                       )

        fine_tune_test_ds = (test_set
                           .shuffle(4096)
                           .map(self.test_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                           .batch(batch_size, drop_remainder=True)
                           .prefetch(1)
                           )
        return fine_tune_ds, fine_tune_test_ds

    def load_dataset(self, test_set_size=4000, shuffle=True):
        examples, metadata = tfds.load('cats_vs_dogs', with_info=True)
        dataset = examples['train']

        if shuffle:
            dataset = dataset.shuffle(8000)

        test_set = dataset.take(test_set_size)
        train_set = dataset.skip(test_set_size)

        return train_set, test_set
