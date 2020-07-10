from classifier.utils import calculate_sigma
from classifier.preprocessor import Preprocessor

# Estimate the sigma scaling factor for a training dataset. Once calculated place in config

preprocessor = Preprocessor()

train_set, _ = preprocessor.load_dataset()

sigma = calculate_sigma(train_set, preprocessor)
print(sigma.numpy())
