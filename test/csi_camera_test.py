from tqdm import tqdm
import torch
import torchvision
import cv2
import numpy as np
import traitlets
from jetbot import Camera, bgr8_to_jpeg, csi_camera
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
#camera = Camera.instance(width=300, height=224, fps=15)
camera = csi_camera.CSI_Camera()

camera.create_gstreamer_pipeline(
sensor_id=0,
#sensor_mode=3,
display_width=300,
display_height=224,
framerate=21,
flip_method=0,
)
camera.open(camera.gstreamer_pipeline)
camera.start()

#動作停止用関数の定義
update_bar(1,'create a function stop demo')
def stop_demo():
    #camera.unobserve(update, names='value')
    camera.stop()
    camera.release()

#カメラupdate時の処理(Main処理定義)
update_bar(1,'create a function that will get called whenever the cameras value changes')

def update(change):
    x = change #['new'] 
    x = preprocess(x) 
    time.sleep(0.048)
        


#処理とカメラの関連付け(処理実行)
update_bar(1,'attach function to the camera for processing')
#camera.observe(update, names='value')  # this attaches the 'update' function to the 'value' traitlet of our camera
_ , camera_image = camera.read()
update(camera_image)  # we call the function once to intialize

update_bar(1,'Complete!')
cnt = 1
t1 = time.time()
try:
    while True:
        _ , camera_image = camera.read()
        update(camera_image)  # we call the function once to intialize
        #time.sleep(1)
        cnt = cnt + 1
        t2 = time.time()
        # framecnt =  camera.frames_read
        if cnt == 100:
            print('frames=', camera.frames_read, '\n', 'isRunnning=', camera.running)
            cnt = 1 
        if  t2 - t1 > 60: 
            break

except KeyboardInterrupt:
    print('Camera stop')

#停止すべき処理
stop_demo()
print('End')


