import cv2
import os
import glob

#获取一张图片的宽高作为视频的宽高
result_path = 'Test_png_clear_diff'
save_path = 'Test_video_clear_diff'
# print((save_path))
# dir_list = glob.glob(result_path+'/*')
# dir_list = [r'E:\ug2+\hard_videos\frame_result\mog2_rgb']

sv_path = save_path#os.path.join(save_path, cls)
# if os.path.exists(sv_path) is False:
#     os.mkdir(sv_path)

# # print(dir_list)
# for k in os.listdir(result_path):
#     dir_list = os.path.join(result_path, k)
#     # cls_dir = os.path.join(result_path, cls)
#     # dir_list = glob.glob(os.path.join(cls_dir, '*'))
#     # break

#     dir = dir_list
#     #print(os.path.exists(dir))
#     # print(dir)
#     imgs = os.listdir(dir)
#     frame_num = len(imgs)
#     # print(frame_num)
    
#     img = os.path.join(dir, imgs[0])
#     # print(img)
#     # print(os.path.exists(img))

#     image=cv2.imread(img)
#     # cv2.imshow("new window", image)   #显示图片
#     image_info=image.shape
#     height=image_info[0]
#     width=image_info[1]
#     size=(height,width)
#     # print(size)
#     # print('dir: ', dir)
#     video_path = os.path.join(sv_path,os.path.basename(dir))
#     #print(video_path)
    
    
#     fps=30
#     fourcc=cv2.VideoWriter_fourcc(*"mp4v")
#     video = cv2.VideoWriter(video_path + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    
#     for i in range(2,frame_num): # front 2, rgb 1
#         file_name = dir + '/' + str(i) + '.png'
#         image=cv2.imread(file_name)
#         #print(image)
#         video.write(image)  # 向视频文件写入一帧--只有图像，没有声音
#         # print('>>>>', i, '/', len(dir_list))
#         # cv2.waitKey()

dir = 'Test_png_clear_diff/2368'
imgs = os.listdir(dir)
frame_num = len(imgs)
# print(frame_num)

img = os.path.join(dir, imgs[0])
# print(img)
# print(os.path.exists(img))

image=cv2.imread(img)
# cv2.imshow("new window", image)   #显示图片
image_info=image.shape
height=image_info[0]
width=image_info[1]
size=(height,width)
# print(size)
# print('dir: ', dir)
video_path = os.path.join(sv_path,os.path.basename(dir))
#print(video_path)


fps=30
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(video_path + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))

for i in range(2,frame_num): # front 2, rgb 1
    file_name = dir + '/' + str(i) + '.png'
    image=cv2.imread(file_name)
    #print(image)
    video.write(image)  # 向视频文件写入一帧--只有图像，没有声音
    # print('>>>>', i, '/', len(dir_list))
    # cv2.waitKey()