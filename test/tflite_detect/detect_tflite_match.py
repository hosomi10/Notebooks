# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
#import tflite_runtime.interpreter as tflite
import numpy as np
from pathlib import Path
from socket import socket, AF_INET, SOCK_DGRAM
import re

DISPLAY_WIDTH = 300
DISPLAY_HEIGHT = 300
colors = ((255, 255, 0), (0, 255, 255), (128, 256, 128), (64, 192, 255), (128, 128, 255)) * 2
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (0,0,255)

HOST = ''   
PORT = 5008

# ソケットを用意
s = socket(AF_INET, SOCK_DGRAM)
# バインドしておく
s.bind((HOST, PORT))
matched_obj = 0

# Pathオブジェクトを生成
p_temp = Path("data/images")

if __name__ == '__main__':
    # load model
    # interpreter = tflite.Interpreter(model_path="model/tflite/detect.tflite")
    interpreter = tf.lite.Interpreter(model_path="data/model/tflite/detect.tflite")
    
    with open('data/model/tflite/labelmap.txt', 'r') as F:
         class_names = F.readlines()

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
while True:
    # 受信
    msg, address = s.recvfrom(8192)
    if msg.decode() == "q":
       print("Sender is closed")
       break
    label = msg.decode().split(':')
    print(label[0])
    matched_obj = int(label[0])
    # except for unlabeled 
    if matched_obj >=1:
    # given label
        check_match = 0
        for p in p_temp.iterdir():
            inImg = str(p) # file only: p.name
            print(inImg)
            if p.is_dir():
               continue

            # prepara input image
            img = cv2.imread(inImg)
            x = cv2.resize(img, (300, 300))
            x = x[:, :, [2,1,0]]  # BGR -> RGB
            x = np.expand_dims(x, axis=0) # 3->4

            # set input tensor
            interpreter.set_tensor(input_details[0]['index'], x)

            # run
            interpreter.invoke()

            # get outpu tensor
            bbox = interpreter.get_tensor(output_details[0]['index'])  # Locations (Top, Left, Bottom, Right)
            class_labels = interpreter.get_tensor(output_details[1]['index'])  # Classes (0=Person)
            Scores = interpreter.get_tensor(output_details[2]['index'])  # Scores
            total_num = interpreter.get_tensor(output_details[3]['index'])  # Number of detections

            dstimg = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            # find matched detect list
            for i in range(int(total_num[0])):
                label = class_labels[0, i].astype(int) # + 1
                class_name = class_names[label].rstrip()
           
                (top, left, bottom, right) = bbox[0, i] * 300
                center_x = (right + left) / 2.0 - 0.5
                center_y = (bottom + top) /2.0 - 0.5
                dis = np.sqrt((150-center_x)**2 + (300-center_y)**2)
                prob = Scores[0, i]
                if prob >= 0.5 and label == matched_obj:
                    print("Class=({},{})".format(i,class_name))
                    print("Probability={}".format(prob))
                    print("Location=({},{})-({},{})".format(int(left), int(top), int(right), int(bottom)))
                    print("Dis={}".format(dis))
                    left = int(left * DISPLAY_WIDTH / 300)
                    right = int(right * DISPLAY_WIDTH / 300)
                    top =  int(top * DISPLAY_HEIGHT / 300)
                    bottom = int(bottom * DISPLAY_HEIGHT / 300)
                    cv2.rectangle(dstimg, (left, top), (right, bottom), colors[i], 1)
                    cv2.circle(dstimg, (int(center_x), int(center_y)), 5, colors[i], -1)
                    cv2.rectangle(dstimg, (left, top+20), (left+160, top), colors[i], -1)
                    cv2.putText(dstimg, "{},{}({:.3f})".format(i,class_name, prob),
                               (left, top+15), font, 0.5, fontColor, 1, cv2.LINE_AA)
                    break

            if prob >= 0.5 and label == matched_obj:
                print("Matched")
                check_match = 1
                break;
        if check_match == 1:
           cv2.imshow("test image", dstimg)
           cv2.moveWindow("test image", 0, 0)
           cv2.waitKey(1500)
           cv2.destroyAllWindows()
        else:
           print('Not matched')
   
# socket close
s.close()
