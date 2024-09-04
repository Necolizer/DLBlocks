import cv2
import os

def video2image(video_dir,save_dir):
    cap = cv2.VideoCapture(video_dir) #生成读取视频对象
    n = 1   #计数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    #获取视频的宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   #获取视频的高度
    fps = cap.get(cv2.CAP_PROP_FPS)    #获取视频的帧率
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))    #视频的编码
    # 定义视频输出
    #writer = cv2.VideoWriter("video_02_result.mp4", fourcc, fps, (width, height))
    i = 0
    timeF = int(fps)     #视频帧计数间隔频率
    while cap.isOpened():
        ret,frame = cap.read() #按帧读取视频
        #到视频结尾时终止
        if ret is False :
            break
        #每隔timeF帧进行存储操作
        # if (n % timeF == 0) :
        i += 1
        print('保存第 %s 张图像' % i)
        save_image_dir = os.path.join(save_dir,'%s.jpg' % i)
        print('save_image_dir: ', save_image_dir)
        cv2.imwrite(save_image_dir,frame, [int( cv2.IMWRITE_JPEG_QUALITY), 100]) #保存视频帧图像
        # n = n + 1
        cv2.waitKey(1) #延时1ms
    cap.release() #释放视频对象

#读取文件夹所有视频，每个视频按帧保存图像
def video2image_multi(video_path,save_path):
    video_list = os.listdir(video_path)
    
    for i in range(len(video_list)) :      
        video_dir = os.path.join(video_path,video_list[i])
        cap = cv2.VideoCapture(video_dir)
        fps = cap.get(cv2.CAP_PROP_FPS)     # 视频的帧率
        save_num = 0
        n = 1           #计数
        timeF = int(fps)          # 视频帧计数间隔频率
        while cap.isOpened():
            ret,frame = cap.read()
            if ret is False :
                break
            #每隔timeF帧进行存储操作
            if (n % timeF == 0) :
                save_num += 1
                save_image_dir = os.path.join(save_path,'%s_%s.jpg' % (i,save_num))
                cv2.imwrite(save_image_dir,frame)
            n = n + 1
            cv2.waitKey(1)
        cap.release()
        print('读取第 %s 个视频完成' % i)

if __name__ == '__main__' :
    video2image(r'D:\PersonalProject\2023-Action2Text\video_samples\S001C001P001R001A055_rgb.avi', 
        r'D:\PersonalProject\2023-Action2Text\video_samples\S001C001P001R001A055_rgb')