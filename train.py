import os
import config
import dataset
import CNN
import save
import cv2
from random import randint
import tensorflow as tf
import pickle as pkl
#training flags

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')

FLAGS = tf.app.flags.FLAGS

def create_checkpoint_dir():
    '''
    Creates the checkpoints directory if it does not exist
    '''
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)


def train(data):

    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[config.img_size_flat], dtype=tf.float32),}
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name

    # x = tf.placeholder(tf.float32, shape=[None, config.img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, config.img_size, config.img_size, config.num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, config.num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    layer_conv1, weights_conv1 = \
    CNN.new_conv_layer(input=x_image,
                   num_input_channels=config.num_channels,
                   filter_size=config.filter_size1,
                   num_filters=config.num_filters1,
                   use_pooling=True)

    layer_conv2, weights_conv2 = \
    CNN.new_conv_layer(input=layer_conv1,
                   num_input_channels=config.num_filters1,
                   filter_size=config.filter_size2,
                   num_filters=config.num_filters2,
                   use_pooling=True)

    layer_conv3, weights_conv3 = \
    CNN.new_conv_layer(input=layer_conv2,
                   num_input_channels=config.num_filters2,
                   filter_size=config.filter_size3,
                   num_filters=config.num_filters3,
                   use_pooling=True)

    layer_flat, num_features = CNN.flatten_layer(layer_conv3)

    layer_fc1 = CNN.new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=config.fc_size,
                         use_relu=True)

    layer_fc2 = CNN.new_fc_layer(input=layer_fc1,
                         num_inputs=config.fc_size,
                         num_outputs=config.num_classes,
                         use_relu=False)   
    y_pred = tf.nn.softmax(layer_fc2,name="y")   
    y_pred_cls = tf.argmax(y_pred, axis=1)

    '''
        layer_fc2  is our predicted probability distribution    -> y
        y_true is the true distribution (the one-hot vector with the digit labels) ->  y_
    '''

    '''
        Training
    '''
    #Note that the function calculates the softmax internally so we must use the output of layer_fc2 directly rather than y_pred which has already had the softmax applied
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true) 
                                                             
    cost = tf.reduce_mean(cross_entropy) # cross_entropy 
    '''
         we ask TensorFlow to minimize cross_entropy(or cost)
    '''
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    values, indices = tf.nn.top_k(y_pred, config.num_classes)

    table = tf.contrib.lookup.index_to_string_table_from_tensor(
      tf.constant([str(i) for i in range(config.num_classes)]))

    prediction_classes = table.lookup(tf.to_int64(indices))

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="correct_pred_mean")

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        CNN.optimize(sess,optimizer,x,y_true,cost,accuracy,data,num_iterations=1)
        CNN.print_validation_accuracy(sess,x,y_true,accuracy,y_pred_cls,data)
        CNN.optimize(sess,optimizer,x,y_true,cost,accuracy,data,num_iterations=99)  # We already performed 1 iteration above.
        CNN.print_validation_accuracy(sess,x,y_true,accuracy,y_pred_cls,data,show_example_errors=True)
        CNN.optimize(sess,optimizer,x,y_true,cost,accuracy,data,num_iterations=900)  # We performed 100 iterations above.
        CNN.print_validation_accuracy(sess,x,y_true,accuracy,y_pred_cls,data,show_example_errors=True)
        # CNN.optimize(sess,optimizer,x,y_true,cost,accuracy,data,num_iterations=9000) # We performed 1000 iterations above.
        # CNN.print_validation_accuracy(sess,x,y_true,accuracy,y_pred_cls,data,show_example_errors=True, show_confusion_matrix=True)

        print('Done training!')
        


        # saver = tf.train.Saver()    
        test_1 = cv2.imread('1228.TIF')
        test_1 = cv2.resize(test_1, (config.img_size, config.img_size), cv2.INTER_LINEAR) / 255
        test_2 = cv2.imread('10889.TIF')
        test_2 = cv2.resize(test_2, (config.img_size, config.img_size), cv2.INTER_LINEAR) / 255
        print("Predicted class for V: {}".format(CNN.sample_prediction(test_1,sess,x,y_true,y_pred_cls)))
        print("Predicted class for Acc: {}".format(CNN.sample_prediction(test_2,sess,x,y_true,y_pred_cls)))
        # saver.save(sess, config.checkpoint_dir+config.model_name +'.ckpt')
        '''
              named_graph_signatures={
                'inputs': exporter.generic_signature({'x': x}),
                'outputs': exporter.generic_signature({'y_pred': layer_fc2})
             }
        '''
         # Export model
        save.export_model(
                    sess,
                    FLAGS.model_version,
                    serialized_tf_example,
                    prediction_classes,
                    values,
                    x,
                    y_pred)
        #Exporting model done
        sess.close()    
        
    

def main():

    # preparations
    create_checkpoint_dir()
    
    data = dataset.read_train_sets(
                config.train_path,
                config.img_size,
                config.classes,
                config.validation_size
                )
    test_data = dataset.read_test_set(
        config.test_path,
        config.img_size
        ).test
    print('Size of: ')
    print("- Training-set:\t\t{}".format(len(data.train.labels)))
    print("- Test-set:\t\t{}".format(len(test_data.ids)))
    print("- Validation-set:\t{}".format(len(data.valid.labels)))

    # Get some random images and their labels from the train set.
    images, cls_true  = data.train.images, data.train.cls

    train(data) #start training data

main()    