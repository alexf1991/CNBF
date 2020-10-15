import os
import numpy as np
from absl import logging
from absl import app
from absl import flags
from utils.utils import *
from utils.trainer import ModelEnvironment
from utils.summary_utils import Summaries
from models.cnbf import CNBF
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
#import keras.backend as K
from loaders.feature_generator import feature_generator
from utils.mat_helpers import *
#from utils.keras_helpers import *
from algorithms.audio_processing import *
from utils.matplotlib_helpers import *
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath('../'))


def preprocess(sample):
    """Preprocess a single sample."""
    return sample


def data_generator(data,batch_size,is_training,is_validation=False,take_n=None,skip_n=None,input_shape=None):
   
    dataset = tf.data.Dataset.from_generator(data,(tf.float32,tf.float32))
    if is_training:
        shuffle_buffer=512

    if skip_n != None:
        dataset = dataset.skip(skip_n)
    if take_n != None:
        dataset = dataset.take(take_n)

    if is_training:

        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size,drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.batch(100)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def learning_rate_fn(epoch):

    if epoch >= 20 and epoch <30:
        return 0.01
    elif epoch >=30 and epoch <40:
        return 0.001
    elif epoch >=40:
        return 0.001
    else:
        return 1.0





# ---------------------------------------------------------
# ---------------------------------------------------------

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '/tmp', 'save directory name')
flags.DEFINE_string('mode', 'local', 'Mode for the training local or cluster')
flags.DEFINE_float('dropout_rate', 0.0, 'dropout rate for the dense blocks')
flags.DEFINE_float('weight_decay', 1e-4, 'weight decay parameter')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('epochs', 40, 'number of epochs')
flags.DEFINE_integer('start_epoch', 0, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', 5, 'Mini-batch size')
flags.DEFINE_boolean('load_model', False, 'Bool indicating if the model should be loaded')
flags.DEFINE_integer('eval_every_n_th_epoch', 1, 'Integer discribing after how many epochs the test and validation set are evaluted')
flags.DEFINE_string('config_file','../cnbf.json','Name of json configuration file')
flags.DEFINE_boolean('predict',False,"Is inference")

def main(argv):
    
    try:

        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

    except KeyError:

        task_id = 0
    
    
    model_save_dir = FLAGS.model_dir
    print("Saving model to : " + str(model_save_dir))
    epochs = FLAGS.epochs
    start_epoch = FLAGS.start_epoch
    dropout_rate = FLAGS.dropout_rate
    weight_decay = FLAGS.weight_decay
    learning_rate = FLAGS.learning_rate
    load_model = FLAGS.load_model
    batch_size = FLAGS.batch_size
    model_save_dir+="_dropout_rate_"+str(dropout_rate)+"_learning_rate_"+str(learning_rate)+"_weight_decay_"+str(weight_decay)

    # Create parameter dict
    params = {}
    params["learning_rate"] = learning_rate
    params["model_dir"] = model_save_dir
    params["weight_decay"] = weight_decay
    params["dropout_rate"] = dropout_rate


    # load config file
    try:
        print('*** loading config file: %s' % FLAGS.config_file)
        with open(FLAGS.config_file, 'r') as f:
            config = json.load(f)
            config["config_file_dir"] = FLAGS.config_file
    except:
        print('*** could not load config file: %s' % FLAGS.config_file)
        quit(0)

    #If load_model get old configuration
    if load_model:
        try:
            params = csv_to_dict(os.path.join(model_dir, "model_params.csv"))
        except:
            print("Could not find model hyperparameters!")

    if FLAGS.predict is False:
        fgen_train = feature_generator(config, 'train')
        fgen_test = feature_generator(config, 'test')
    else:
        fgen_test = feature_generator(config, 'test')

    input_shape = (batch_size, fgen_train.nfram, fgen_train.nbin, fgen_train.nmic)
    #ResNet 18
    model = CNBF(config = config,
                 fgen = fgen_train,
                 batch_size=batch_size,
                 name = "cnbf",
                 kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                 kernel_initializer=tf.keras.initializers.he_normal(),
                 dropout=dropout_rate)

    #Train data generator
    train_ds = data_generator(fgen_train.generate,batch_size,is_training=True,input_shape = input_shape)
    steps_train = fgen_train.steps
    #Test data generator
    test_ds = data_generator(fgen_test.generate,100,is_training=False,input_shape = input_shape)
    #Create summaries to log
    scalar_summary_names = ["total_loss",
                            "bf_loss",
                            "weight_decay_loss",
                            "accuracy"]

    summaries = Summaries(scalar_summary_names = scalar_summary_names,
                          learning_rate_names = ["learning_rate"],
                          save_dir = model_save_dir,
                          modes = ["train","test"],
                          summaries_to_print={"train": ["total_loss", "accuracy"],
                                              "eval":["total_loss", "accuracy"]})

    #Create training setttings for models
    model_settings = [{'model': model,
            'optimizer_type': tf.keras.optimizers.Adam,
            'base_learning_rate': learning_rate,
            'learning_rate_fn': learning_rate_fn,
            'init_data': [tf.random.normal(input_shape),tf.random.normal(input_shape)],
            'trainable':True}]
    
    #Write training configuration into .csv file
    write_params_csv(model_save_dir, params)

    # Build training environment
    trainer = ModelEnvironment(train_ds,
                               None,
                               test_ds,
                               epochs,
                               EvalFunctions,
                               model_settings=model_settings,
                               summaries=summaries,
                               eval_every_n_th_epoch = 1,
                               num_train_batches=steps_train,
                               load_model=load_model,
                               save_dir = model_save_dir,
                               input_keys=[0,1],
                               label_keys=[],
                               start_epoch=start_epoch)
    
    trainer.train()

if __name__ == '__main__':
  app.run(main)
  
