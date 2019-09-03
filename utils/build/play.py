#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-13 23:09:37
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$

import os
import cv2

img_path = r"../data/"
number = input("Please input number: ")
img_path = img_path + number

for root, dirs, files in os.walk(img_path):
	for file in files:
		if file.endswith('.png'):
			img = cv2.imread(os.path.join(root, file))
			cv2.imshow("gif", img)
			cv2.waitKey(100)
