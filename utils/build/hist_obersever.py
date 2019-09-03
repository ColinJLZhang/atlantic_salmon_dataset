#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-22 19:11:52
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$

import os
import re
import cv2
import pylab as plt

#按照顺序返回文件夹中的图像帧的绝对路径
def getimgList(path):
	imgL = [img for img in os.listdir(path) if os.path.splitext(img)[-1] == '.png']
	#获得列表顺序不一定是按照截取的帧名称排序现在进行排序,按照名字中的数字排序
	imgL = sorted(imgL, key=lambda img: eval(re.search(r'\d+',img).group()))
	imgL = [os.path.join(path,imgname) for imgname in imgL]
	return imgL

def plotHist(imglist):
	plt.figure()
	for step in imglist:
		for imgname in step:
			img = cv2.imread(imgname,0)
			hist = cv2.calcHist([img], [0], None, [256], [0.0,255.0])
			plt.plot(list(range(len(hist))),hist, label=os.path.split(imgname)[-1])
	plt.title('frame-hist')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	path = r'E:\python_practice_code\image processing\academic\texture_video_movement'
	imglist = getimgList(path)
	print('list:',imglist)
	num = 8 #每幅图的显示图例
	splitImaglist = []
	for i in range(int(len(imglist)/num)+1):
		temp = [imglist[:num],]
		splitImaglist.append(temp)
		imglist = imglist[num:]
	for imgi in splitImaglist:
		plotHist(imgi)
