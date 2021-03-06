from tqdm import tqdm
from jetbot import Robot
import time

#progress barの定義
bar = tqdm(total = 2)

def update_bar(update_val,display_text):
    bar.update(update_val)
    bar.set_description(display_text)
    
#robotインスタンス生成
update_bar(1,'create our robot instance')
robot = Robot()

#動作停止用関数の定義
def stop_demo():
    robot.stop()

update_bar(1,'Complete!')

#ロボット分岐用カウンタ
c = 0

try:
    while True:
        c = c + 1
        if c%2 == 1:
            robot.forward(0.3)
        else:
            robot.left(0.3)
        time.sleep(3)
        print('processing')

except KeyboardInterrupt:
    print('motor stop')
    #停止すべき処理
    stop_demo()

