#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-19 23:08:16
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$

import numpy as np
import cv2
from PIL import ImageEnhance
from PIL import Image
import matplotlib.pyplot as plt
import os 
import pickle

root = r"D:\1dataset"
name = "GH096787.MP4"
# TODO: 去除在视频中增氧部分的帧
# 3min51s 增氧结束 GH096787
path = os.path.join(root, name)

# 彩色图直方图均衡
def hisEqulColor(img):
    imgnew = img.copy()
    ycrcb = cv2.cvtColor(imgnew, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    # print(len(channels))
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, imgnew)
    return imgnew

# 亮度增强，使用Image类
def enhance_brightness(image, bright=1.5):
    image = image.copy()
    image = Image.fromarray(image)
    enh_bri = ImageEnhance.Brightness(image)
    brightness = bright
    image_brightened = enh_bri.enhance(brightness)
    return np.array(image_brightened)

# 彩色图像直方图，实时显示当前帧直方图
def color_hist(img1, img2=None):
    chans1 = cv2.split(img1)
    chans2 = cv2.split(img2)
    colors = ('b', 'g', 'r')
 
    plt.subplot(2,1,1)
    plt.cla()
    plt.title("Color Histogram")
    plt.xlabel("gray Level")
    plt.ylabel("number of pixels") 
    for (chan, color) in zip(chans1, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.axis([0, 256, 0, 8000])
        plt.plot(hist, color = color, label= color)
        plt.xlim([0, 256])
        plt.legend()  
    
    plt.subplot(2,1,2)
    plt.cla()
    plt.xlabel("gray Level")
    plt.ylabel("number of pixels")    
    for (chan, color) in zip(chans2, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])        
        plt.axis([0, 256, 0, 8000])
        plt.plot(hist, color = color, label= color)
        plt.xlim([0, 256])
        plt.legend()    

############################################################################################
# 按一定间隔从视频中截取图片，以帧为单位。所有截取的图片为灰并存在一个np.array                    #     
# 一个样本包含了5s 150frame ， 每个图片为一个np.array，所有的array组成一个list                 #
# 同时每个样本生成了一个pickle文件                                                           #
# 图片命名规则：{VideoName}_{#SampleNumber}_{#FrameNumber}.png, "GH076787_105_145.png"      #
############################################################################################
def frame2img(videopath, time):
    cap = cv2.VideoCapture(videopath)
    # cap.set(0, 231000)  # 从231s开始正常的鱼类运动图像，单位毫秒
    print("Video info:\n width:{}, height:{}, fps:{}".format(cap.get(3), cap.get(4), cap.get(5)))
    pkl_data =[]
    # ret = True
    # TODO: 第27帧莫名其妙， 每次在第27帧， ret, frame = cap.read() ret的结果是False, frame的结果是None
    while cap.isOpened():
        count = int(cap.get(1))        
        ret, frame = cap.read()
        # 获取当前的帧数
        if frame is not None:
            if count % time == 0:
                print(count, end = " =>>")
                frame = cv2.resize(frame, (int(frame.shape[1] * 0.2), int(frame.shape[0] * 0.2)), interpolation=cv2.INTER_CUBIC)
                frame.astype(int)
                img_enhance = enhance_brightness(frame, 1.5)
                pkl_data.append(img_enhance)

                # 绘制直方图
                plt.ion()
                color_hist(frame, img_enhance)            
                plt.pause(0.000005)
                imgpath = '..\\imgs\\none-eating\\{}_{}_{}.png'.format(name.split('.')[0], count//150, count%150)
                cv2.imwrite(imgpath, img_enhance)
                
                # pickle 存数据
                if count % 150 == 0:
                    pklpath = '..\\pkl\\none-eating\\{}_{}_.pkl'.format(name.split('.')[0], count//150)
                    f = open(pklpath, 'wb')
                    pickle.dump(pkl_data, f)
                    pkl_data = []
                
                # 显示帧             
                cv2.imshow("before_video",frame)
                cv2.imshow("after_video",img_enhance)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    print(os.getcwd())
    frame2img(path, 1)
