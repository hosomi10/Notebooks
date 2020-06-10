# -*- coding: utf-8 -*-
import cv2
import io
import time
import numpy as np
from PIL import Image
# import tensorflow as tf
import tflite_runtime.interpreter as tflite
from pathlib import Path
# from socket import socket, AF_INET, SOCK_DGRAM

DISPLAY_WIDTH = 300
DISPLAY_HEIGHT = 300
colors = ((255, 255, 0), (0, 255, 255), (128, 256, 128), (64, 192, 255), (128, 128, 255)) * 2
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (0,0,255) # B G R

# socket HOST
# HOST = ''
# PORT = 5008
# ADDRESS = "172.19.39.123" # 自分に送信 127.0.0.1

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  # set input tensor
  interpreter.set_tensor(tensor_index, image)
  #input_details = interpreter.get_input_details()
  #output_details = interpreter.get_output_details()
  #input_tensor = interpreter.tensor(tensor_index)()[0]
  #input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke() # run
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)
  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

if __name__ == '__main__':
 
    # inception_label.txt
    labels = load_labels('model/tflite/labels_mobilenet_quant_v1_224.txt') 

    # load model
    # for tensorflow 1.12
    # interpreter = tf.contrib.lite.Interpreter(model_path="detect.tflite")
    # 1.13 lite: contrib -> core move
    # inceptionV3.tflite
    interpreter = tflite.Interpreter(model_path="model/tflite/mobilenet_v1_1.0_224_quant.tflite")
    interpreter.allocate_tensors()

    input_shape = interpreter.get_input_details()[0]['shape']
    input_type = interpreter.get_input_details()[0]['dtype']
    width = input_shape[1]
    height = input_shape[2]
    print(input_type)
    [1 ]
# init socket
# s = socket(AF_INET, SOCK_DGRAM)

p_temp = Path("test_image")

for p in p_temp.iterdir():
    inImg = str(p) # file only: p.name
    print(inImg)
    # prepara input image
    img = cv2.imread(inImg)
    # x = cv2.resize(img, (width, height))
    # x = x[:, :, [2,1,0]].astype(input_type)  # BGR -> RGB
    # x = np.expand_dims(x, axis=0) # 3->4
    image = Image.open(inImg).convert('RGB').resize((width, height),Image.ANTIALIAS)
    x = np.asarray(image).astype(input_type)
    x = np.expand_dims(x, axis=0)

    results = classify_image(interpreter, x)
    label_id, prob = results[0]
  
    print("predicted number is {} [{:.2f}]".format(labels[label_id], prob))

    #send messages
    # msg = p.name + ':' + labels[label_id]
    # s.sendto(msg.encode(), (ADDRESS, PORT))

    # display the image & result at window
    dstimg = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    # cv2.rectangle(dstimg, (100, 25), (400, 5), colors[2], -1)
    # cv2.putText(dstimg, "{} ({:.3f})".format(labels[label_id], prob),
               # (100, 20), font, 0.6, fontColor, 1, cv2.LINE_AA)
    cv2.imshow("test image", dstimg)
    cv2.moveWindow("test image", 0, 0)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()

# msg = "q"
# s.sendto(msg.encode(), (ADDRESS, PORT))
# s.close()