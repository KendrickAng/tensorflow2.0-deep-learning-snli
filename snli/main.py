# Implementation of Recurrent Neural Networks with LSTMs for Question-Answering

import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import logging
import pprint # pretty print python objects
import sys

pp = pprint.PrettyPrinter(indent=2)
FLAGS = None
logger = None

def main():
    global FLAGS
    global logger

    '''
    STEP 1: LOAD HYPERPARAMETERS
    '''
    logger = create_logger()
    FLAGS = load_config()
    logger.info("FLAGS: \n{}".format(pp.pformat(vars(FLAGS)))) # vars returns __dict__ attribute
    '''
    STEP 2: LOAD AND SPLIT DATASET
    '''
    train_dataset, validate_dataset, test_dataset = load_datasets()
    debug_print_dataset_item(train_dataset)

def load_datasets():
    snli_dataset, snli_info = tfds.load(name='snli', split=None, with_info=True)
    snli_train = snli_dataset["train"]
    snli_validate = snli_dataset["validation"]
    snli_test = snli_dataset["test"]
    # shuffle with buffer    
    BUFFER_SIZE = FLAGS.buffer_size
    shuffled_snli_train = snli_train.shuffle(BUFFER_SIZE)
    shuffled_snli_validate = snli_validate.shuffle(BUFFER_SIZE)
    shuffled_snli_test = snli_test.shuffle(BUFFER_SIZE)
    # batch datasets    
    BATCH_SIZE = FLAGS.batch_size
    train_dataset = shuffled_snli_train.batch(BATCH_SIZE)
    validate_dataset = shuffled_snli_validate.batch(tf.cast(snli_info.splits["validation"].num_examples, tf.int64))
    test_dataset = shuffled_snli_test.batch(tf.cast(snli_info.splits["test"].num_examples, tf.int64))
    return train_dataset, validate_dataset, test_dataset

def create_logger():
    log = logging.getLogger() # root logger
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="%(asctime)s : %(levelname)s %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return logging.getLogger(__name__)

def load_config():
    logger.info("TENSORFLOW IN EAGER MODE: {}".format(tf.executing_eagerly()))
    parser = argparse.ArgumentParser("Hyperparameters for SNLI textual entailment")
    parser.add_argument("--config_file", default="snli.config", help="path to config file")
    sys.stdout.flush() # flush buffer into terminal
    flags, _ = parser.parse_known_args()
    # read from config file and transfer to FLAGS
    import json
    with open(flags.config_file, 'r') as f:
        config = json.load(f) # turn json encoded data to python object
        for key in config:
            setattr(flags, key, config[key])
    return flags # namespace object

"""
Prints an item from the dataset. If the ds is batched, prints a batch.
"""
def debug_print_dataset_item(dataset):
    global pp
    for item in dataset.take(tf.constant(1, dtype=tf.int64)): # FeaturesDict
        #logger.debug(item)
        logger.debug("DATA ARRAY: {}".format(pp.pformat(item['hypothesis'].numpy())))

if __name__ == "__main__":
    main()
