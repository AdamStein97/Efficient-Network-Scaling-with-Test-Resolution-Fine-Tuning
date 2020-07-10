import os
import tensorflow as tf
import classifier as c
from classifier.preprocessor import Preprocessor
from classifier.trainer import Trainer
from classifier.utils import load_config
from classifier.tf_models.classifier_model import ImageClassifier

config = load_config(master_config_name="fine_tune_config.yaml")

preprocessor = Preprocessor(**config)

train_set, test_set = preprocessor.load_dataset(**config)

train_ds, test_ds = preprocessor.make_finetune_datasets(**config)

model = ImageClassifier(**config)

inp = tf.zeros((64, 64, 64, 3))
_ = model(inp)

model.load_weights(os.path.join(c.MODEL_DIR, "trained_classifier"))

trainer = Trainer(model=model, **config)

model = trainer.train(train_ds, test_ds)

model.save_weights(os.path.join(c.MODEL_DIR, "fine_tuned"))