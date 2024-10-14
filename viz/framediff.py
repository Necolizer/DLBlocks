import numpy as np
import cv2
import os
from glob import glob
from time import time


def MOG2Frame(root_dir, vid_sample, save_dir):

    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, vid_sample[:-4])
    # print(vid_sample[:-4])
    # print(save_dir)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    else:
        return
    vid = os.path.join(root_dir, vid_sample)

    cap = cv2.VideoCapture(vid)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(frame_count, height, width)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    ret, frame = cap.read()
    pre_frame = frame
    cnt = 0
    while(1):
        
        cnt += 1

        frame_diff = cv2.absdiff(frame, pre_frame)

        frame_diff = (frame_diff - frame_diff.min()) / (frame_diff.max() - frame_diff.min()) * 255


        if frame_diff is not None:
            # print((fgmask.shape))
            cv2.imwrite(os.path.join(save_dir, str(cnt)+".png"), frame_diff)
            # video.write(fgmask)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        pre_frame = frame.copy()
        ret, frame = cap.read()


        if ret is False:
            break
        # if cnt == frame_count:
        #     break
        
        
    cap.release()

if __name__ == '__main__':
    root_dir = 'Test_video_clear_new'
    save_dir = 'Test_png_clear_diff'

    vid = 'Test_video_clear_new/2368.mp4'
    MOG2Frame(root_dir, vid[len(root_dir)+1:], save_dir)