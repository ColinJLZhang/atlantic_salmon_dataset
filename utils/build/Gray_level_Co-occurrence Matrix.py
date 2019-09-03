#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-27 10:09:13
# @Author  : https://blog.csdn.net/qq_23926575/article/details/80599199
# @Link    : http://darklunar.ml
# @Version : $Id$

import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

def glcm(arr, d_x, d_y, gray_level=16):
    '''计算并返回归一化后的灰度共生矩阵'''
    max_gray = arr.max() + 1
    print(max_gray)
    if max_gray < gray_level:
    	gray_level = max_gray
    #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    height,width = arr.shape
    
#灰度共生矩阵
    # #cv2.imread出现的数据类型为uint8的数据类型，在做乘法运算时会超出限制，所以必须转换数据类型
    # arr = arr.astype(float)
    # arr = arr * gray_level // max_gray
    # arr = arr.astype(int)
    # ret = np.zeros([gray_level, gray_level])
    # for j in range(height - d_y):
    #     for i in range(width - d_x):
    #          rows = arr[j][i]
    #          cols = arr[j + d_y][i + d_x]
    #          ret[rows][cols] += 1.0
    # #把共生矩阵转换成对称矩阵，即（2，3）与（3，2）算作同一共生值
    # ret = (np.triu(ret)+np.tril(ret).T).T+np.triu(ret,1)+np.tril(ret,-1).T
    # return ret / float(height * width]) # 归一化

#灰度梯度共生矩阵
    gsx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
    gsy = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
    height, width = arr.shape
    grad = (gsx ** 2 + gsy ** 2) ** 0.5 # 计算梯度值
    grad = np.asarray(1.0 * grad * (gray_level-1) / grad.max(), dtype=np.int16)
    gray = np.asarray(1.0 * arr * (gray_level-1) / arr.max(), dtype=np.int16) # 0-255变换为0-15
    gray_grad = np.zeros([gray_level, gray_level]) # 灰度梯度共生矩阵
    for i in range(height):
        for j in range(width):
            gray_value = gray[i][j]
            grad_value = grad[i][j]
            gray_grad[gray_value][grad_value] += 1
    gray_grad = 1.0 * gray_grad / (height * width) # 归一化灰度梯度矩阵，减少计算量
    return gray_grad


if __name__ == '__main__':
	fp = r"E:\python_practice_code\image processing\academic\texture_video_movement\frame 6000.png"
	img = cv2.imread(fp,0)
	# img = cv2.resize(img,(960,540),interpolation=cv2.INTER_CUBIC)
	# plt.imshow(img)
	# plt.show()
	# #test
	# img = np.array([[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]])
	# print(img)
	glcm_0 = glcm(img,20,0,16)
	print(glcm_0)
	#绘制三维图
	from mpl_toolkits.mplot3d import Axes3D
	X = np.arange(len(glcm_0))
	Y = np.arange(len(glcm_0))
	X,Y = np.meshgrid(X,Y)
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.plot_surface(X,Y,glcm_0, rstride=1, cstride=1, cmap='rainbow')
	plt.figure()
	plt.imshow(glcm_0)
	plt.show()