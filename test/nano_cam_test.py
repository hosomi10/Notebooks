import cv2
import numpy as np

GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)15/1 \
    ! nvvidconv ! video/x-raw, width=(int)1640, height=(int)1232, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
WINDOW_NAME = 'Camera Test'

def main():
    cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)

    while True:
        ret, img = cap.read()
        
        w = int(300)
        h = int(224) 
        dst = cv2.resize(img, (w, h))
        dst1 = cv2.resize(img, (224,224))
        if ret != True:
            break

        key = cv2.waitKey(10)
        if key == 97: # a resize
            print('resize image')
            #cv2.imwrite('origin_img.jpg', img)
            cv2.imwrite('nano224_img.jpg', dst1)
            cv2.imwrite('resize300_img.jpg', dst)

        if key == 98: # b clip
            print('crop') 
            dst2 = dst[0:224, 38:262]
            cv2.imwrite('clip_img.jpg', dst2)

        cv2.imshow(WINDOW_NAME, dst)

        if key == 27: # ESC 
            break

if __name__ == "__main__":
    main()
