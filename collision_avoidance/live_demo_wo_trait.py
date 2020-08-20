#progress barの定義
from tqdm import tqdm
from threading import Lock
bar = tqdm(total = 8)
softmax_lock = Lock()

#2020/8/6 input(runtime) -> time[s] for live demo
import sys
print('How long do you want to run this program? [s]')
runtime=input()

if runtime.isnumeric() == True:
    runtime = int(runtime)
else:
    print('input error start program again')
    sys.exit()

def update_bar(update_val,display_text):
    bar.update(update_val)
    bar.set_description(display_text)
    #print(display_text)

#Pytorch modelの初期化
bar.set_description('live demo')
print('initialize pytorch model')

#Pillow 7.0.0 のエラー回避
import PIL
PIL.PILLOW_VERSION=PIL.__version__

import torch
import torchvision

model = torchvision.models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)


#modelの読み込み
update_bar(1,'load model')

model.load_state_dict(torch.load('best_model.pth'))

#GPUに転送
update_bar(1,'transfer to the GPU device')

device = torch.device('cuda')
model = model.to(device)

#事前処理関数の作成
update_bar(1,'Create the preprocessing function')

import cv2
import numpy as np

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

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

#カメラ起動、画面表示(pyではwidgetが使えないため代替案検討)
update_bar(1,'start and display our camera')

#import traitlets
#from IPython.display import display
#import ipywidgets.widgets as widgets
from jetbot import bgr8_to_jpeg, csi_camera

#use csi camera for image
#camera = Camera.instance(width=300, height=224, fps=15)
###new process###
csi_cam = csi_camera.CSI_Camera()
csi_cam.create_gstreamer_pipeline(
sensor_id=0,
#sensor_mode=3,
capture_width=3280,
capture_height=2464,
display_width=300,
display_height=224,
framerate=21,
flip_method=0,
)

csi_cam.open(csi_cam.gstreamer_pipeline)
csi_cam.start()
#image = widgets.Image(format='jpeg', width=224, height=224)
#blocked_slider = widgets.FloatSlider(description='blocked', min=0.0, max=1.0, orientation='vertical')
#camera_link = traitlets.dlink((camera, 'value'), (image, 'value'), transform=bgr8_to_jpeg)
#display(widgets.HBox([image, blocked_slider]))

#robotインスタンス生成
update_bar(1,'create our robot instance')

from jetbot import thread_robot

robot = thread_robot.Robot()
robot.start()

"""
#tkinter生成
import tkinter
"""

#動作停止用関数の定義
def stop_demo():
    #camera.unobserve(update, names='value')
    csi_cam.stop()
    csi_cam.release()
    time.sleep(1)
    robot.stop()

#カメラupdate時の処理(Main処理定義)
update_bar(1,'create a function that will get called whenever the cameras value changes')

import torch.nn.functional as F
import time
def update(change):
    global blocked_slider, robot
    softmax_lock.acquire()
    x = change #['new'] 
    x = preprocess(x)    
    # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
    y = model(x) 
    y = F.softmax(y, dim = 1)
    prob_blocked = float(y.flatten()[0])

    if prob_blocked < 0.6:
        robot.forward(0.3)
    else:
        robot.left(0.3)
        #capture debug 
        #cv2.imwrite('wo_trait_block.jpg', change[0:224, 38:262])
        #cv2.imwrite('wo_trait_without_clip.jpg', change)
        #robot.stop()
        #print('please press Ctrl + C in 60sec')
        #time.sleep(60)
    softmax_lock.release()
    
    time.sleep(0.047)

#処理とカメラの関連付け(処理実行)
update_bar(1,'attach function to the camera for processing')

#camera.observe(update, names='value')  # this attaches the 'update' function to the 'value' traitlet of our camera
_ , csi_image=csi_cam.read() 
update(csi_image) # we call the function once to intialize
robot.stop()

update_bar(1,'Complete!')

print('Press Enter key to start')
dummy=input()
print('press Ctrl+C to stop robot')
t1 = time.time()
try:
    cnt = 0
    while True:
        t2 = time.time()
        _ , csi_image=csi_cam.read()
        update(csi_image)
        
        if cnt % 100 == 0:
            print("frames=", csi_cam.frames_read)
        cnt += 1        
        #time.sleep(5)
        #print('processing')
        if t2 - t1 > runtime:
            print("End by time")
            break

except KeyboardInterrupt:
    print('Robot stop')
#停止すべき処理
#need camera and motor proces release
cnt = 0
robot.stop()
stop_demo()

#GUIを表示し続ける
#root.mainloop()
