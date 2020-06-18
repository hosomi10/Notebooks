# -*- coding: utf-8 -*-
import cv2
# import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
from pathlib import Path
import time

DISPLAY_WIDTH = 300
DISPLAY_HEIGHT = 300
colors = ((255, 255, 0), (0, 255, 255), (128, 256, 128), (64, 192, 255), (128, 128, 255)) * 2
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (0,0,255)

mean = 255.0 * np.array([0.5, 0.5, 0.5])
stdev = 255.0 * np.array([0.5, 0.5, 0.5])

def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])

def bgr8_to_tf_input(img):
    #x = cv2.resize(img, (300, 300))
    x = img
    x = x[:, :, [2,1,0]]  # BGR -> RGB
    x = np.expand_dims(x, axis=0) # 3->4
    return x

def listup_files(path):
    yield [os.path.abspath(p) for p in glob.glob(path)]

if __name__ == '__main__':
    # load model
    # for tensorflow 1.12
    # interpreter = tf.contrib.lite.Interpreter(model_path="detect.tflite")
    #interpreter = tf.lite.Interpreter(model_path="model/tflite/detect.tflite")
    # using tflite runtime
    interpreter = tflite.Interpreter(model_path="../../model/tflite_model/detect.tflite")

    with open('../../model/tflite_model/labelmap.txt', 'r') as F:
         class_names = F.readlines()

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# Pathオブジェクトを生成
p_temp = Path("../../data/300")

for p in p_temp.iterdir():
    inImg = str(p) # file only: p.name
    print(inImg)
    if p.is_dir():
       continue

    # prepara input image
    img = cv2.imread(inImg)
    test_x = bgr8_to_tf_input(img)
    
    start_time = time.monotonic()
    # set input tensor
    interpreter.set_tensor(input_details[0]['index'], test_x)

    # run
    interpreter.invoke()

    # get outpu tensor
    tflite_results1 = interpreter.get_tensor(output_details[0]['index'])  # Locations (Top, Left, Bottom, Right)
    tflite_results2 = interpreter.get_tensor(output_details[1]['index'])  # Classes (0=Person)
    tflite_results3 = interpreter.get_tensor(output_details[2]['index'])  # Scores
    tflite_results4 = interpreter.get_tensor(output_details[3]['index'])  # Number of detections

    elapsed_ms = (time.monotonic() - start_time) * 1000
    print('detect time={}'.format(elapsed_ms))

    # Display detect results to the image(stream)
    dstimg = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    for i in range(int(tflite_results4[0])):
        (top, left, bottom, right) = tflite_results1[0, i] * 300
        #class_name = class_names[tflite_results2[0, i].astype(int) + 1].rstrip() # label unlabel start
        class_name = class_names[tflite_results2[0, i].astype(int)].rstrip() # label person start
        prob = tflite_results3[0, i]
        if prob >= 0.5:
            print("Location=({},{})-({},{})".format(int(left), int(top), int(right), int(bottom)))
            print("Class={}".format(class_name))
            print("Probability={}".format(prob))
            left = int(left * DISPLAY_WIDTH / 300)
            right = int(right * DISPLAY_WIDTH / 300)
            top =  int(top * DISPLAY_HEIGHT / 300)
            bottom = int(bottom * DISPLAY_HEIGHT / 300)
            cv2.rectangle(dstimg, (left, top), (right, bottom), colors[i], 1)
            cv2.rectangle(dstimg, (left, top+20), (left+160, top), colors[i], -1)
            cv2.putText(dstimg, "{} ({:.3f})".format(class_name, prob),
                       (left, top+15), font, 0.5, fontColor, 1, cv2.LINE_AA)
        # else:
            # print("Low Prob or Not Recognized={}".format(i))

    cv2.imshow("test image", dstimg)
    cv2.moveWindow("test image", 0, 0)
    # print result
    #result = np.argmax(probs[0])
    #score = probs[0][result]
    #print("predicted number is {} [{:.2f}]".format(result, score))

    cv2.waitKey(2000)
    cv2.destroyAllWindows()
