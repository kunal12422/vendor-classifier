import tensorflow as tf
import numpy as np
import dataset


def main():
    # create an empty graph for the session
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        saver = tf.model.import_meta_graph(config.checkpoint_dir+config.model_name +'.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint(config.checkpoint_dir))

        # get necessary tensors by name

        