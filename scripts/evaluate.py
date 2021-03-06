import os
import numpy as np
from absl import logging
from absl import app
from absl import flags
from utils.utils import *
from utils.trainer import ModelEnvironment
from utils.summary_utils import Summaries
from models.cnbf import CNBF
from models.complex_cnn import CNBF_CNN
from models.eval_functions.nbf_loss import EvalFunctions
from loaders.feature_generator import feature_generator
import time
import numpy as np
import argparse
import json
import os
import sys

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Activation, LSTM, Input, Lambda
# import keras.backend as K
from loaders.feature_generator import feature_generator
from utils.mat_helpers import *
# from utils.keras_helpers import *
from algorithms.audio_processing import *
from utils.matplotlib_helpers import *
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath('../'))


def preprocess(sample):
    """Preprocess a single sample."""
    return sample


def data_generator(data, batch_size, is_training, is_validation=False, take_n=None, skip_n=None, input_shape=None):
    dataset = tf.data.Dataset.from_generator(data, (tf.float32, tf.float32))
    if is_training:
        shuffle_buffer = 64

    if skip_n != None:
        dataset = dataset.skip(skip_n)
    if take_n != None:
        dataset = dataset.take(take_n)

    if is_training:

        # dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def learning_rate_fn(epoch):
    if epoch >= 20 and epoch < 30:
        return 0.01
    elif epoch >= 30 and epoch < 40:
        return 0.001
    elif epoch >= 40:
        return 0.001
    else:
        return 1.0


# ---------------------------------------------------------
# ---------------------------------------------------------

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '/tmp', 'save directory name')
flags.DEFINE_string('mode', 'local', 'Mode for the training local or cluster')
flags.DEFINE_integer('start_epoch', 0, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', 5, 'Mini-batch size')
flags.DEFINE_integer('eval_every_n_th_epoch', 1,
                     'Integer discribing after how many epochs the test and validation set are evaluted')
flags.DEFINE_string('config_file', '../cnbf.json', 'Name of json configuration file')



def main(argv):
    try:

        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

    except KeyError:

        task_id = 0

    model_save_dir = FLAGS.model_dir
    print("Saving model to : " + str(model_save_dir))
    start_epoch = FLAGS.start_epoch
    load_model = True
    batch_size = FLAGS.batch_size

    # load config file
    try:
        print('*** loading config file: %s' % FLAGS.config_file)
        with open(FLAGS.config_file, 'r') as f:
            config = json.load(f)
            config["config_file_dir"] = FLAGS.config_file
            config["predictions_path"] = os.path.join(FLAGS.model_dir,"predictions")
            if not(os.path.exists(config["predictions_path"])):
                os.makedirs(config["predictions_path"])
    except:
        print('*** could not load config file: %s' % FLAGS.config_file)
        quit(0)

    # If load_model get old configuration
    if load_model:
        try:
            params = csv_to_dict(os.path.join(model_save_dir, "model_params.csv"))
        except:
            print("Could not find model hyperparameters!")


    fgen_test = feature_generator(config, 'test', steps=batch_size)

    input_shape = (batch_size, fgen_test.nfram, fgen_test.nbin, fgen_test.nmic)
    # ResNet 18
    model = CNBF_CNN(config=config,
                     fgen=fgen_test,
                     n_ch_base=8,
                     batch_size=batch_size,
                     name="cnbf",
                     kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                     kernel_initializer=tf.keras.initializers.he_normal(),
                     dropout=0.0)

    # Test data generator
    test_ds = data_generator(fgen_test.generate, batch_size, is_training=False, input_shape=input_shape)
    # Create summaries to log
    scalar_summary_names = ["total_loss",
                            "bf_loss",
                            "weight_decay_loss",
                            "accuracy"]

    summaries = Summaries(scalar_summary_names=scalar_summary_names,
                          learning_rate_names=["learning_rate"],
                          save_dir=model_save_dir,
                          modes=["train", "test"],
                          summaries_to_print={"train": ["total_loss", "accuracy"],
                                              "eval": ["total_loss", "accuracy"]})

    # Create training setttings for models
    model_settings = [{'model': model,
                       'optimizer_type': tf.keras.optimizers.Adam,
                       'base_learning_rate': 1e-3,
                       'learning_rate_fn': learning_rate_fn,
                       'init_data': [tf.random.normal(input_shape), tf.random.normal(input_shape)],
                       'trainable': True}]

    # Build training environment
    env = ModelEnvironment(None,
                               None,
                               test_ds,
                               0,
                               EvalFunctions,
                               model_settings=model_settings,
                               summaries=summaries,
                               eval_every_n_th_epoch=1,
                               num_train_batches=1,
                               load_model=load_model,
                               save_dir=model_save_dir,
                               input_keys=[0, 1],
                               label_keys=[],
                               start_epoch=start_epoch)
    results = []
    for data in test_ds:
        res = env.predict(data,training=False)
        results.append(res["predictions"])


if __name__ == '__main__':
    app.run(main)

