from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from MIRNet.mirnet.inference import Inferer
from MIRNet.mirnet.utils import download_dataset, plot_result

import tensorflow as tf
import numpy as np

#download_dataset for testing (run only once)
#download_dataset('LOL')

LOW_LIGHT_IMGS = glob('./eval15/low/*')

MODEL_DICT = {
    "dr": "mirnet_dr.tflite",
    "fp16": "mirnet_fp16.tflite",
    "int8": "mirnet_int8.tflite"
}

def preprocess_image(image_path, image_resize_factor=1):
    original_image = Image.open(image_path)
    width, height = original_image.size
    preprocessed_image = original_image.resize(
        (
            width // image_resize_factor,
            height // image_resize_factor
        ),
        Image.ANTIALIAS)
    preprocessed_image = tf.keras.preprocessing.image.img_to_array(preprocessed_image)
    preprocessed_image = preprocessed_image.astype('float32') / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    return original_image, preprocessed_image

def infer_tflite(model_type, image):
    interpreter = tf.lite.Interpreter(model_path=MODEL_DICT[model_type])
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    interpreter.resize_tensor_input(0, [1, image.shape[1], image.shape[2], 3])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    raw_prediction = interpreter.tensor(output_index)
    output_image = raw_prediction()

    output_image = output_image.squeeze() * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    output_image = Image.fromarray(np.uint8(output_image))
    return output_image

model_type = "fp16" #@param ["dr", "fp16", "int8"]

for image_path in LOW_LIGHT_IMGS[:5]:
    original_image, preprocessed_image = preprocess_image(image_path)
    output_image = infer_tflite(model_type, preprocessed_image)
    plot_result(Image.fromarray(np.uint8(original_image)), output_image)
