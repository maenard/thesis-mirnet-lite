from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from MIRNet.mirnet.inference import Inferer
from MIRNet.mirnet.utils import download_dataset, plot_result
import os

import tensorflow as tf
import numpy as np
LOW_LIGHT_IMGS = glob('./eval15/low/*')

MODEL_DICT = {
    "dr": "mirnet_dr.tflite",
    "fp16": "mirnet_fp16.tflite",
    "int8": "mirnet_int8.tflite"
}

def infer_tflite(model_type, image_path):
    original_image = Image.open(image_path)
    image = original_image.resize((input_width, input_height), Image.ANTIALIAS)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter = tf.lite.Interpreter(model_path=MODEL_DICT[model_type])
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    interpreter.resize_tensor_input(0, [1, image.shape[1], image.shape[2], 3])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    raw_prediction = interpreter.tensor(output_index)()
    
    output_image = raw_prediction.squeeze() * 255.0
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_image)
    
    return original_image, output_image

# Create a directory to save the output images
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

model_type = "int8" #dr, fp16, int8
input_width, input_height = 64, 64

for i, image_path in enumerate(LOW_LIGHT_IMGS[:5]):
    original_image, output_image = infer_tflite(model_type, image_path)
    
    # Save the output image to the output directory
    output_path = os.path.join(output_dir, f"output_{i}.jpg")
    output_image.save(output_path)

print("Output images saved in the 'output_images' directory.")
