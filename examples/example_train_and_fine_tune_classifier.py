import os
import classifier as c
from classifier.preprocessor import Preprocessor
from classifier.trainer import Trainer
from classifier.utils import load_config

# Train classifier
config = load_config()

preprocessor = Preprocessor(**config)

trainer = Trainer(**config)

train_set, test_set = preprocessor.load_dataset(**config)

train_ds, test_ds = preprocessor.make_train_datasets(**config)

model = trainer.train(train_ds, test_ds)


# Finetune to test resolution
fine_tune_config = load_config(master_config_name="fine_tune_config.yaml")

fine_tune_trainer = Trainer(model=model, **fine_tune_config)

fine_tune_train_ds, fine_tune_test_ds = preprocessor.make_finetune_datasets(**fine_tune_config)

model = fine_tune_trainer.train(fine_tune_train_ds, fine_tune_test_ds)

model.save_weights(os.path.join(c.MODEL_DIR, "fully_trained_classifier"))