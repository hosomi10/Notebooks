from tkinter import *
from tkinter import ttk
from socket import socket, AF_INET, SOCK_DGRAM

HOST = ''
PORT = 5008
ADDRESS = "127.0.0.1" # 自分に送信 

def stop_click():
    msg = "q"
    s.sendto(msg.encode(), (ADDRESS, PORT))
    s.close()

def send_click():
    show_selection()
    
def listbox_selected(event):
    show_selection()
    
def show_selection():
    for i in lb.curselection():
        #send messages
        msg = str(lb.get(i)) # 
        s.sendto(msg.encode(), (ADDRESS, PORT))
        print(msg)

if __name__ == '__main__':
    root = Tk()
    root.title('Label Listbox')
    
    # init socket
    s = socket(AF_INET, SOCK_DGRAM)

    # フレーム
    frame1 = ttk.Frame(root, padding=10)
    frame1.grid()
    # リストボックス    
    labels = ('detect_list')
    v1 = StringVar(value=labels)
    lb = Listbox(frame1, listvariable=v1,height=20)

    with open('data/model/coco_labels.txt', 'r') as F:
         class_names = F.readlines()
         

    for i, name in enumerate(class_names):
        #lb.insert(END, class_names[i].rstrip())
        labels = str(i)+':'+ name
        if i >=1:
           lb.insert(END, labels)

    lb.bind('<<ListboxSelect>>', listbox_selected)
    lb.grid(row=0, column=0)
    
    #Button
    button1 = ttk.Button(frame1, text='Send', command=send_click)
    button1.grid(row=1, column=0)
    button2 = ttk.Button(frame1, text='Stop', command=stop_click)
    button2.grid(row=2, column=0)
    
    root.mainloop()