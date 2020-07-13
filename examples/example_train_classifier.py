import os
import classifier as c
from classifier.preprocessor import Preprocessor
from classifier.trainer import Trainer
from classifier.utils import load_config

config = load_config()

preprocessor = Preprocessor(**config)

trainer = Trainer(**config)

train_set, test_set = preprocessor.load_dataset(**config)

train_ds, test_ds = preprocessor.make_train_datasets(train_set, test_set, **config)

model = trainer.train(train_ds, test_ds)

model.save_weights(os.path.join(c.MODEL_DIR, "classifier"))