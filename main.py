
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import numpy as np

from mpl_toolkits.mplot3d.axes3d import Axes3D

import scipy
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os
import glob

images1 = glob.glob('/Users/sm/Dropbox (VR Holding BV)/de-vignetting/set1/DPP_b4after/orig*.*')
images1.sort()
images2 = glob.glob('/Users/sm/Dropbox (VR Holding BV)/de-vignetting/set1/DPP_b4after/devegn*.*')
images2.sort()

def sample(diff):
    X = []
    Y = []
    Z = []
    for r in range(0, diff.shape[0], 10):
        for c in range(0, diff.shape[1], 10):
            X.append(r)
            Y.append(c)
            Z.append(diff[r,c])
    X = np.asarray(X).reshape((int(diff.shape[0] / 10 + 1), int(diff.shape[1] / 10 + 1)))
    Y = np.asarray(Y).reshape((int(diff.shape[0] / 10 + 1), int(diff.shape[1] / 10 + 1)))
    Z = np.asarray(Z).reshape((int(diff.shape[0] / 10 + 1), int(diff.shape[1] / 10 + 1)))
    return X, Y, Z

_ , img_extension = os.path.splitext(images1[0])

i = 0
for fname in images1:
    dir_path = os.path.dirname(fname)
    i += 1

    # load the two input images
    ImgOrig = cv2.imread(fname)
    head, _  = os.path.split(dir_path)
    #fname2 = head + '/full_devignetted/' + os.path.basename(fname)
    fname2 = '%s/devegn%.2d%s' % (dir_path, i, img_extension)
    ImgDevegn = cv2.imread(fname2)

    # resizeFactor = 0.2
    # ImgOrig = cv2.resize(ImgOrig, None, fx=resizeFactor, fy=resizeFactor, interpolation=cv2.INTER_CUBIC)
    # ImgDevegn = cv2.resize(ImgDevegn, None, fx=resizeFactor, fy=resizeFactor, interpolation=cv2.INTER_CUBIC)


    if 1 :
        ImgOrigTra = cv2.cvtColor(ImgOrig, cv2.COLOR_BGR2Lab)
        ImgDevegnTra = cv2.cvtColor(ImgDevegn, cv2.COLOR_BGR2Lab)

        ImgOrigTra = ImgOrigTra[:, :, 0].astype('float')
        ImgDevegnTra = ImgDevegnTra[:, :, 0].astype('float')
        mxA = ImgOrigTra.max()
        minA = ImgOrigTra.min()

        # V =  ImgDevegnTra - ImgOrigTra
        # V = cv2.divide(ImgOrigTra, ImgDevegnTra)
        V = np.asarray(ImgDevegnTra, dtype=np.floating) - np.asarray(ImgOrigTra, dtype=np.floating)

        # mx = V.max()
        # mi = V.min()
        VNormalized = cv2.normalize(V.astype('float'), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # Convert to normalized floating point
        # mxn = VNormalized.max()
        # min = VNormalized.min()
        #coeff = (coeff * 255).astype("uint8")

    if 0 :
        #imageAL, imageAa, imageAb = cv2.split(imageALab)
        ImgOrigTra = ImgOrig[:, :, 0].astype('float')
        ImgDevegnTra = ImgDevegn[:, :, 0].astype('float')

        #V = cv2.divide(ImgDevegnTra, ImgOrigTra )
        V = ImgDevegnTra - ImgOrigTra
        mx = V.max()
        mi = V.min()
        VNormalized = cv2.normalize(V.astype('float'), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # Convert to normalized floating point

        mxn = VNormalized.max()
        min = VNormalized.min()
        #coeff = (coeff * 255).astype("uint8")

    if 0 :
        height, width, channels = ImgOrig.shape

        #xx = ["%.2d" % i for i in range(10,20)]
        #xx = [int("%.2d" % i ) for i in range(10,20)]
        xx = list(range(int(height/2), height)) #numpy.arange(height/2, height, 2)
        yy = list(range(int(width/2), width))
        pixelsX = ImgOrigTra[xx, int(width/2)]
        pixelsY = ImgOrigTra[int(height/2), yy]

        plt.plot(pixelsX)
        plt.ylabel('fall-off')
        plt.savefig('/Users/sm/Documents/PycharmProjects/devignetting/results/pixelsX%.2d.JPG' % i, bbox_inches='tight')
        plt.show()

    # plt.imshow(V)
    cv2.imwrite('/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/ImgOrig%.2d.JPG' % i, ImgOrigTra)
    cv2.imwrite('/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res//ImgDevegn%.2d.JPG' % i, ImgDevegnTra)
    cv2.imwrite('/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/VNormalized%.2d.JPG' % i, VNormalized)


    # plot 3d
    if 0 :
        x, y = np.mgrid[0:VNormalized.shape[0], 0:VNormalized.shape[1]]
        ax = plt.gca(projection='3d')
        ax.clear()
        ax.plot_surface(x, y, VNormalized)
        plt.savefig('/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/pixelsX%.2d.JPG' % i, bbox_inches='tight')

    if 1 :
        X_d, Y_d, Z_d = sample(V)
        fig = plt.figure(figsize=(14, 6))

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        p = ax.plot_surface(X_d, Y_d, Z_d, rstride=1, cstride=1, cmap="coolwarm", linewidth=0, antialiased=True)
        plt.savefig('/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/pixelsX%.2d.JPG' % i, bbox_inches='tight')



cv2.destroyAllWindows()