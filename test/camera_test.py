from tqdm import tqdm
import torch
import torchvision
import cv2
import numpy as np
import traitlets
from jetbot import Camera, bgr8_to_jpeg
import time

#progress barの定義
bar = tqdm(total = 6)

def update_bar(update_val,display_text):
    bar.update(update_val)
    bar.set_description(display_text)

#事前処理関数の作成
update_bar(1,'Create the preprocessing function')

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
device = torch.device('cuda')
normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value[0:224, 38:262]
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x
 
#カメラ起動
update_bar(1,'start and display our camera')
camera = Camera.instance(width=300, height=224, fps=15)

#動作停止用関数の定義
update_bar(1,'create a function stop demo')
def stop_demo():
    camera.unobserve(update, names='value')
    camera.stop()

#カメラupdate時の処理(Main処理定義)
update_bar(1,'create a function that will get called whenever the cameras value changes')

def update(change):
    x = change['new'] 
    x = preprocess(x) 
    time.sleep(0.001)
        
update({'new': camera.value})  # we call the function once to intialize

#処理とカメラの関連付け(処理実行)
update_bar(1,'attach function to the camera for processing')
camera.observe(update, names='value')  # this attaches the 'update' function to the 'value' traitlet of our camera

update_bar(1,'Complete!')

try:
    while True:
        time.sleep(5)
        print('processing')
        #key = cv2.waitKey(10)
        #if key == 27: #escape key
            #print('esc pressed')
            #break

except KeyboardInterrupt:
    print('Camera stop')

#停止すべき処理
stop_demo()
print('End')
quit()

