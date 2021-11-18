##########################################
# Convert ".h5" model to ".tflite" model
##########################################

import tensorflow as tf

image_shape = (1,224,224,3) # Values may change depending on the model being used.

def representative_data_gen():
    num_calibration_images = 100
    for _ in range(num_calibration_images):
        image = tf.random.normal([image_shape[0], image_shape[1],image_shape[2], 3])
        yield [image]


model = tf.keras.models.load_model('fas.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizing the model for smaller size and faster inference
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Make the model to TPU-Compatible format
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_opssu = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert model to TFLite format
tflite_model = converter.convert()
open("test_fas_optimized_uint8.tflite", "wb").write(tflite_model)

# Show the model type of the converted model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)




