#progress barの定義
from tqdm import tqdm
bar = tqdm(total = 8)

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

import traitlets
#from IPython.display import display
#import ipywidgets.widgets as widgets
from jetbot import Camera, bgr8_to_jpeg

camera = Camera.instance(width=300, height=224)
#image = widgets.Image(format='jpeg', width=224, height=224)
#blocked_slider = widgets.FloatSlider(description='blocked', min=0.0, max=1.0, orientation='vertical')
#camera_link = traitlets.dlink((camera, 'value'), (image, 'value'), transform=bgr8_to_jpeg)
#display(widgets.HBox([image, blocked_slider]))

#robotインスタンス生成
update_bar(1,'create our robot instance')

from jetbot import Robot

robot = Robot()

#tkinter生成
import tkinter

#動作停止用関数の定義
def stop_demo():
    camera.unobserve(update, names='value')
    robot.stop()

#GUIの作成
if __name__ == "__main__":
    root = tkinter.Tk()
    
    #変数の設定
    var = tkinter.DoubleVar(
        master=root,
        value=0.500,
    )

    #現在値表示用ラベルの設定
    l = tkinter.Label(
        master=root,
        width=50,
        textvariable=var,
    )
    l.pack()

    #スケールの設定
    s = tkinter.Scale(
        master=root,
        orient="horizon",
        showvalue=False,
        variable=var,
        from_=0.0,
        to=1.0,
        resolution=0.001,
        length=200,
        )
    s.pack()

    #ボタンの設定
    b = tkinter.Button(
        text='Stop　JETBOT',
        width=20,
        command=stop_demo,
        )
    b.pack()

#カメラupdate時の処理(Main処理定義)
update_bar(1,'create a function that will get called whenever the cameras value changes')

import torch.nn.functional as F
import time

def update(change):
    global blocked_slider, robot
    x = change['new'] 
    x = preprocess(x)
    y = model(x) 
    
    # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
    y = F.softmax(y, dim = 1)
    
    prob_blocked = float(y.flatten()[0])
    
    var.set(prob_blocked)
    
    if prob_blocked < 0.6:
        robot.forward(0.3)
    else:
        robot.left(0.3)
    
    time.sleep(0.001)
        
update({'new': camera.value})  # we call the function once to intialize

#処理とカメラの関連付け(処理実行)
update_bar(1,'attach function to the camera for processing')

camera.observe(update, names='value')  # this attaches the 'update' function to the 'value' traitlet of our camera

update_bar(1,'Complete!')

#GUIを表示し続ける
root.mainloop()
