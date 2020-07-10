import tensorflow as tf
import time
import datetime
import os

import classifier as c
from classifier.tf_models.classifier_model import ImageClassifier

class Trainer():
    def __init__(self, epochs=150, log=True, save=True, restore=False, lr=1e-3, log_name="classifier_log",
                 model=None, model_save_directory="trainer_checkpoints", **kwargs):
        self.epochs = epochs
        self.log = log
        self.save = save

        if model is None:
            self.model = ImageClassifier(**kwargs)
        else:
            self.model = model

        self.optimizer = tf.keras.optimizers.Adam(lr)

        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory=os.path.join(c.MODEL_DIR, model_save_directory), max_to_keep=5)

        if restore:
            checkpoint.restore(self.manager.latest_checkpoint)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Accuracy(name='train_acc')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Accuracy(name='val_acc')

        if self.log:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(c.LOG_DIR, log_name + current_time + '/train')
            val_log_dir = os.path.join(c.LOG_DIR, log_name + current_time + '/val')
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    @staticmethod
    def loss_function(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))

    @tf.function
    def train_step(self, img, y_true):
        with tf.GradientTape() as tape:
            y_pred = self.model(img)
            loss = self.loss_function(y_true, y_pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        label_pred = tf.argmax(y_pred, axis=-1)

        self.train_loss(loss)
        self.train_accuracy(y_true, label_pred)

    @tf.function
    def eval_step(self, img, y_true):
        y_pred = self.model(img, training_mbconv=False, training_classification_layers=False)
        loss = self.loss_function(y_true, y_pred)

        label_pred = tf.argmax(y_pred, axis=-1)

        self.val_loss(loss)
        self.val_accuracy(y_true, label_pred)

    def train(self, train_ds, test_ds):
        for epoch in range(self.epochs):
            start = time.time()
            for batch, (image, label) in enumerate(train_ds):
                self.train_step(image, label)

                if batch % 10 == 0:
                    print('Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        batch, self.train_loss.result(), self.train_accuracy.result()))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, self.train_loss.result(), self.train_accuracy.result()))

            for batch, (image, label) in enumerate(test_ds):
                self.eval_step(image, label)

            print('Epoch {} Val Loss {:.4f} Val Accuracy {:.4f}'.format(
                epoch + 1, self.val_loss.result(), self.val_accuracy.result()))

            if self.log:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)
                with self.val_summary_writer.as_default():
                    tf.summary.scalar('val_loss', self.val_loss.result(), step=epoch)
                    tf.summary.scalar('val_accuracy', self.val_accuracy.result(), step=epoch)

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

            if (epoch + 1) % 10 == 0 and self.save:
                self.manager.save()

        return self.model

