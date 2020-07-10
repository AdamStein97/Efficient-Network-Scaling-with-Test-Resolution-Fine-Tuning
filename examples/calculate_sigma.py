from classifier.utils import calculate_sigma
from classifier.preprocessor import Preprocessor


preprocessor = Preprocessor()

train_set, _ = preprocessor.load_dataset()

sigma = calculate_sigma(train_set, preprocessor)
print(sigma.numpy())
