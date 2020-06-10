# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np

DISPLAY_WIDTH = 480
DISPLAY_HEIGHT = 480
colors = ((255, 255, 0), (0, 255, 255), (128, 256, 128), (64, 192, 255), (128, 128, 255)) * 2
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (0,0,255)

if __name__ == '__main__':
    # prepara input image
    img = cv2.imread('test_image/elephant3.jpg')
    x = cv2.resize(img, (300, 300))
    x = x[:, :, [2,1,0]]  # BGR -> RGB
    x = np.expand_dims(x, axis=0) # 3->4

    # load model
    # for tensorflow 1.12
    # interpreter = tf.contrib.lite.Interpreter(model_path="detect.tflite")
    interpreter = tf.lite.Interpreter(model_path="model/tflite/detect.tflite")
    
    with open('model/tflite/labelmap.txt', 'r') as F:
         class_names = F.readlines()

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input tensor
    interpreter.set_tensor(input_details[0]['index'], x)

    # run
    interpreter.invoke()

    # get outpu tensor
    tflite_results1 = interpreter.get_tensor(output_details[0]['index'])  # Locations (Top, Left, Bottom, Right)
    tflite_results2 = interpreter.get_tensor(output_details[1]['index'])  # Classes (0=Person)
    tflite_results3 = interpreter.get_tensor(output_details[2]['index'])  # Scores
    tflite_results4 = interpreter.get_tensor(output_details[3]['index'])  # Number of detections

    dstimg = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    for i in range(int(tflite_results4[0])):
        (top, left, bottom, right) = tflite_results1[0, i] * 300
        class_name = class_names[tflite_results2[0, i].astype(int) + 1].rstrip()
        prob = tflite_results3[0, i]
        if prob >= 0.6:
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

    cv2.imshow("test image", dstimg)
    cv2.moveWindow("test image", 0, 0)
    # print result
    #result = np.argmax(probs[0])
    #score = probs[0][result]
    #print("predicted number is {} [{:.2f}]".format(result, score))

    cv2.waitKey(0)
    cv2.destroyAllWindows()