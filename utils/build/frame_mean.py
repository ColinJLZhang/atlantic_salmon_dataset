#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-24 15:45:29
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$

import numpy as np
import os
import re
import cv2
import pylab as plt
from hist_obersever import getimgList

def calulate_mean_frame(imglist):
	frameMean = []
	for imgname in imglist:
		img = cv2.imread(imgname,0)
		frameMean.append(img.mean())
	np.savetxt('frame-mean.txt',np.array(frameMean))
	return frameMean

def calulate_mean_video(path):
	frameMean = []
	cap = cv2.VideoCapture(path)
	while cap.isOpened():
		ret,frame = cap.read()
		if frame is None:
			break
		else:
			imgEntroy = 0
			frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			img = cv2.resize(frame,(960,540),interpolation=cv2.INTER_CUBIC)
			frameMean.append(img[120:,:].mean())#把水表面的光去除
			print(cap.get(1))
	cap.release()
	np.savetxt('g34-cut-video-mean.txt',np.array(frameMean))
	return frameMean





if __name__ == '__main__':
	# framePath = r'E:\python_practice_code\image processing\academic\texture_video_movement'
	# imglist = getimgList(path)
	# frameMean = calulate_mean_frame(imglist)
	videoPath = r"E:\研究生\渔机所\烟台东方海洋出差\实验视频及图片\GOPR6434.MP4"
	frameMean = calulate_mean_video(videoPath)
	plt.figure()
	plt.plot(list(range(len(frameMean))),frameMean)
	plt.title('g34-video-mean')
	plt.savefig('g34-cut-video-mean.svg')
	plt.show()