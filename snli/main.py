# Implementation of Recurrent Neural Networks with LSTMs for Question-Answering

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import argparse
import logging
import pprint # pretty print python objects
import sys
import os

pp = pprint.PrettyPrinter(indent=2)
FLAGS = None
logger = None
glove_wordmap = {}
glove_wordmap_size = 0

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
    '''
    STEP 3: LOAD GLOVE EMBEDDINGS
    '''
    prepare_glove_embeddings()
    load_glove_embeddings()
    '''
    STEP 4: DEFINE MODEL
    '''
    max_hypothesis_length = FLAGS.max_hypothesis_length # max no. words in a question
    embed_size = FLAGS.embedding_size # how big is each word vector
    max_features = FLAGS.max_features # how many unique words to use (num rows in embedding vector?)    
    max_premise_length = FLAGS.max_premise_length
    batch_size = FLAGS.batch_size
    hidden_length = FLAGS.hidden_length
    num_epochs = FLAGS.num_epochs

    import tensorflow.keras.layers as tfl
    model = tf.keras.Sequential()
    model.add(tfl.Bidirectional(tfl.LSTM(hidden_length, return_sequences=True), input_shape=(batch_size, max_hypothesis_length, embed_size))) 
    model.add(tfl.Bidirectional(tfl.LSTM(hidden_length, return_sequences=False))) 
    model.add(tfl.Dense(3, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.summary()

    model.fit(x=train_dataset, epochs=num_epochs, validation_data=validate_dataset, validation_steps=5)

def sentence2sequence(sentence):
    '''
    Turns an input sentence into a (n, d) matrix. 
    n is the number of tokens in the sentence.
    d is the number of dimensions each word vector has.
    '''
    tokens = None

    try:
        tokens = sentence.decode().lower()
    except AttributeError: # not byte-encoded
        tokens = sentence.lower()
    rows = []
    words = []
    for token in tokens: # each token is a word in the sentence
        i = len(token)
        while len(token) > 0 and i > 0:
            word = token[:i]
            if word in glove_wordmap:
                rows.append(glove_wordmap[token])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                # no such word, add keep reducing until we find a word
                i = i - 1
    return rows, words

def load_glove_embeddings():
    global glove_wordmap
    global glove_wordmap_size

    glove_text_file = FLAGS.word_embeddings_txtfilename
    printOne = True    

    with open(glove_text_file, "r") as glove:
        for line in glove:
            values = line.split()
            word = values[0]
            # tensorflow only accepts arrays, not python lists
            featuresMatrix = np.asarray(values[1:], dtype='float32')
            # print a sample word with feature matrix
            if printOne:
                printOne = False
                logger.info("Sample word \"{}\" with features {}".format(word, pp.pformat(featuresMatrix)))
            glove_wordmap[word] = featuresMatrix
    glove_wordmap_size = len(glove_wordmap)
    logger.info("Glove wordmap populated, found %s vectors" % glove_wordmap_size)

def prepare_glove_embeddings():
    glove_link = FLAGS.word_embeddings_link
    glove_zip_file = FLAGS.word_embeddings_zipfilename
    glove_text_file = FLAGS.word_embeddings_txtfilename
    
    if (not os.path.isfile(glove_zip_file) and not os.path.isfile(glove_text_file)):
        logger.info("Glove embeddings not found. Downloading from site...")
        import urllib.request
        # download glove zip file
        print(glove_link)
        print(glove_zip_file)
        urllib.request.urlretrieve(glove_link, glove_zip_file)
        logger.info("Glove embeddings file downloaded.")
        # extract zip to text file
        unzip_single_file(glove_zip_file, glove_text_file)
    return

def unzip_single_file(zip_file_name, output_file_name):
    """
    If the outfile exists, don't recreate, else create from zipfile
    """
    if not os.path.isfile(output_file_name):
        import zipfile
        logger.info("Unzipping glove embeddings {}..".format(zip_file_name))
        with open(output_file_name, "wb") as out_file:
            with zipfile.ZipFile(zip_file_name) as zipped:
                for info in zipped.infolist():
                    if output_file_name in info.filename:
                        with zipped.open(info) as requested_file:
                            out_file.write(requested_file.read())
                            logger.info("Glove embeddings unzipped to {}".format(output_file_name))
                            return
    return

def transform_dataset(tf_dataset):
    return ([tf_dataset['premise'], tf_dataset['hypothesis']], 
            tf_dataset['label'])

def load_datasets():
    printOne = True

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
    # (input sentence, hypothesis), train_labels
    train_dataset, train_labels = shuffled_snli_train.batch(BATCH_SIZE).map(transform_dataset)
    logger.warning("HELP ME {}{}".format(train_dataset, train_labels))
    validate_dataset = shuffled_snli_validate.batch(tf.cast(snli_info.splits["validation"].num_examples, tf.int64))
    test_dataset = shuffled_snli_test.batch(tf.cast(snli_info.splits["test"].num_examples, tf.int64))
    
    logger.info("BATCHED DATASET: {}".format(validate_dataset))
    logger.info("BATCHED DATASET PIECE: {}".format(next(iter(validate_dataset))))
    if printOne:
        printOne = False
        debug_print_dataset_item(train_dataset)

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
    parser.add_argument("--buffer_size", type=int, default=10000, help="images to include when shuffling dataset")
    parser.add_argument("--batch_size", type=int, default=100, help="training set size")
    parser.add_argument("--word_embedding_link", type=str, default="http://nlp.stanford.edu/data/glove.6B.zip", help="link to glove word embeddings")
    parser.add_argument("--word_embedding_zipfilename", type=str, default="glove.6B.zip", help="filename to save glove word embeddings .zip")
    parser.add_argument("--word_embedding_txtfilename", type=str, default="glove.6B.50d.txt", help="filename to save glove word embeddings .txt")
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
        logger.debug("DATASET INPUT ARRAY (PREMISE): \n{}".format(item['premise'].numpy()[0]))
        logger.debug("DATASET INPUT ARRAY (HYPOTHESIS): \n{}".format(item['hypothesis'].numpy()[0]))
        logger.debug("DATASET INPUT ARRAY (LABEL): \n{}".format(item['label'].numpy()[0]))
        # logger.debug("DATA ARRAY: {}".format(pp.pformat(item['hypothesis'].numpy())))

if __name__ == "__main__":
    main()
