import tensorflow as tf
import os

tf.random.set_seed(1)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
MODEL_CONFIG_DIR = os.path.join(CONFIG_DIR, "model_config")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
LOG_DIR = os.path.join(ROOT_DIR, "logs")