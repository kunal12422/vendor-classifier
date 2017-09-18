import tensorflow as tf
import os
import config
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat

#Save Model
def export_model(sess,model_version,serialized_tf_example,prediction_classes,values,x,y_pred):
    
    export_path = os.path.join(
        compat.as_bytes(config.trained_model),
        compat.as_bytes(str(model_version)))
    print('Exporting trained model to %s' % export_path)
    builder = saved_model_builder.SavedModelBuilder(export_path)
    classification_inputs = utils.build_tensor_info(serialized_tf_example)
    classification_outputs_classes = utils.build_tensor_info(prediction_classes)
    classification_outputs_scores = utils.build_tensor_info(values)
    classification_signature = signature_def_utils.build_signature_def(
        inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
        outputs={
            signature_constants.CLASSIFY_OUTPUT_CLASSES:
                classification_outputs_classes,
            signature_constants.CLASSIFY_OUTPUT_SCORES:
                classification_outputs_scores
            },
        method_name=signature_constants.CLASSIFY_METHOD_NAME)
            
    tensor_info_x = utils.build_tensor_info(x)

    tensor_info_y = utils.build_tensor_info(y_pred)

    prediction_signature = signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x},
        outputs={'scores': tensor_info_y},
        method_name=signature_constants.PREDICT_METHOD_NAME)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    
    #add the sigs to the servable
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
            },
        legacy_init_op=legacy_init_op)
    #save it!
    builder.save()



    # init_op = tf.group(tf.initialize_all_tables(), name='init_op')
    # saver = tf.train.Saver(sharded=True)
    # model_exporter = exporter.Exporter(saver)
    # model_exporter.init(
    #     sess.graph.as_graph_def(),
    #      init_op=init_op,
    #      named_graph_signatures={
    #         'inputs': exporter.generic_signature({'x': x}),
    #         'outputs': exporter.generic_signature({'y_pred': layer_fc2})
    #      }
    # )
    # model_exporter.export(config.trained_model, tf.constant(FLAGS.export_version), sess)


    print('Done exporting!')