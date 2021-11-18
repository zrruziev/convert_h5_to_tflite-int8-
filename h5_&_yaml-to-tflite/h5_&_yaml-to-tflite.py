################################################################################
# Convert Keras model with two inputs to ".tflite" (compatible with Coral-TPU)
################################################################################
import tensorflow as tf
from tensorflow.keras.models import model_from_yaml
from os.path import splitext

image_shape = (1,128,128,3)
rppg_shappe = (1,3)
def representative_data_gen():
    num_calibration_images = 200
    for _ in range(num_calibration_images):
        image = tf.random.normal([image_shape[0], image_shape[1], image_shape[2], 3])
        rppg = tf.random.normal([rppg_shappe[0], rppg_shappe[1]])
        yield [image, rppg]


def load_model(path,custom_objects={},verbose=0):
    path = splitext(path)[0]
    with open('%s.yaml' % path,'r') as yaml_file:
        model_yaml = yaml_file.read()
    model = tf.keras.models.model_from_yaml(model_yaml, custom_objects=custom_objects)
    model.load_weights('%s.h5' % path)
    if verbose: print('Loaded from %s' % path)
    return model

# Load the model
mod_path = "rppg-model.h5"
keras_mod = load_model(mod_path)
converter = tf.lite.TFLiteConverter.from_keras_model(keras_mod)

# Optimizing the model for smaller size and faster inference
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Make the model compatible with Coral-TPU
converter.representative_dataset = representative_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert model to TFLite format
tflite_model = converter.convert()
# Save the TF Lite model.
with tf.io.gfile.GFile('test_rppg-optimized-uint8.tflite', 'wb') as f:
    f.write(tflite_model)

# Show the type of the converted model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

