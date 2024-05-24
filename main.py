import cv2

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

subsample = 20
def build3Dmesh(diff):
    h, w = diff.shape
    R = range(0, h, subsample)
    C = range(0, w, subsample)
    X = np.zeros((len(R), len(C)))
    Y = np.zeros((len(R), len(C)))
    Z = np.zeros((len(R), len(C)))
    for r in range(0 , len(R)):
        for c in range(0 , len(C)):
            X[r,c] = r
            Y[r,c] = c
            Z[r,c] = diff[R[r],C[c]]
    return X, Y, Z

def getCirclePixels(x0, y0, R, imH, imW):
    # sample points
    theta = np.linspace(0.1, 2 * np.pi-0.1, 2048)  # make it finer for finer circles
    # the pixels that get hit
    xy = [xy for xy in zip( ( - R * np.sin(theta) + x0).astype(int), (R * np.cos(theta) + y0).astype(int) ) if xy[0] >= 0 and xy[0] < imH and xy[1] >= 0 and xy[1] < imW]
    return np.array(xy)

method = "Blue" # extracting vignetting image from any of RGB chanels results in the same model, i.e. PS applies the same model to all three chanells

cross = "vertical"

set = "synthetic"
set = "test"
set = "set3"
set = "test2"

info = "_rec709_g1"
info = "_rec709_g0.45"
info = "_adobeRGB_g0.45"
info = "_adobeRGB_g1"

VdppList = []
VpsList = []
minList = []
maxList = []

for imgNo in range(1,2) :
    #photoshop 32bit output is float32
    orig = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/synthetic/g%.2d.tif" % imgNo, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # orig = orig * 65535 #float32 (0-1) to uint16 (0-65535)
    # orig = orig.astype(np.uint16)

    #ACR output is uint16
    PS0d = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/%s/PS_b4after/0d%.2d.jpg" % (set, imgNo), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    PS100d = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/%s/PS_b4after/100d%.2d.jpg" % (set, imgNo), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    DPP0d = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/%s/DPP_b4after/0d%.2d.jpg" % (set, imgNo), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    DPP100d = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/%s/DPP_b4after/100d%.2d.jpg" % (set, imgNo), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    mx1 = DPP100d.max()
    mi1 = DPP100d.min()

    if method == "Blue":  # Blue from BGR
        origGreen = orig[:, :, 1].astype('float').copy()

        DPP0dBlue = DPP0d[:, :, 0].astype('float').copy()
        DPP0dGreen = DPP0d[:, :, 1].astype('float').copy()
        DPP0dRed = DPP0d[:, :, 2].astype('float').copy()
        DPP100dBlue = DPP100d[:, :, 0].astype('float').copy()
        DPP100dGreen = DPP100d[:, :, 1].astype('float').copy()
        DPP100dRed = DPP100d[:, :, 2].astype('float').copy()

        mx1 = PS0d[:, :, 0].max()
        mi1 = PS0d[:, :, 0].min()

        PS0dBlue = PS0d[:, :, 0].astype('float').copy()

        mx1 = PS0dBlue.max()
        mi1 = PS0dBlue.min()

        PS0dGreen = PS0d[:, :, 1].astype('float').copy()
        PS0dRed = PS0d[:, :, 2].astype('float').copy()
        PS100dBlue = PS100d[:, :, 0].astype('float').copy()
        PS100dGreen = PS100d[:, :, 1].astype('float').copy()
        PS100dRed = PS100d[:, :, 2].astype('float').copy()

        Vdpp = np.asarray(DPP100dGreen, dtype=np.double) / (np.asarray(DPP0dGreen, dtype=np.double) + 0)
        Vps = np.asarray(PS100dBlue, dtype=np.double) / (np.asarray(PS0dBlue, dtype=np.double) + 0)

    elif method == "Lab": # L from Lab
        origLab = cv2.cvtColor(orig, cv2.COLOR_BGR2Lab) #!! 16bits images are not supported
        origL = origLab[:, :, 0].astype('float')

        DPP0dLab = cv2.cvtColor(DPP0d, cv2.COLOR_BGR2Lab)
        DPP100dLab = cv2.cvtColor(DPP100d, cv2.COLOR_BGR2Lab)
        DPP0dL = DPP0dLab[:, :, 0].astype('float')
        DPP100dL = DPP100dLab[:, :, 0].astype('float')

        PS0dLab = cv2.cvtColor(PS0d, cv2.COLOR_BGR2Lab)
        PS100dLab = cv2.cvtColor(PS100d, cv2.COLOR_BGR2Lab)
        PS0dL = PS0dLab[:, :, 0].astype('float')
        PS100dL = PS100dLab[:, :, 0].astype('float')

        Vdpp = np.asarray(DPP100dL, dtype=np.double) / (np.asarray(DPP0dL, dtype=np.double) + 0)
        Vps = np.asarray(PS100dL, dtype=np.double) / (np.asarray(PS0dL, dtype=np.double) + 0)

    # plt.imshow(Vdpp)
    maxList.append(Vdpp.max())
    minList.append(Vdpp.min())

    VdppList.append(Vdpp)
    VpsList.append(Vps)

    VNormalized = cv2.normalize(Vdpp, None, 0, 255, cv2.NORM_MINMAX)  # Convert to normalized floating point
    # VNormalized = Vdpp * 100
    cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/%s/VNormalized%s_%.2d.TIF" % (set, info, imgNo), VNormalized)

    if 1:
        height, width = Vdpp.shape

        xx = list(range(0, height))  # numpy.arange(height/2, height, 2)
        yy = list(range(0, width))
        xy  = getCirclePixels(int(height / 2), int(width / 2), int(width / 2.3), height, width)

        if method == "Blue":
            if cross == "vertical" :
                h, w = origGreen.shape
                xxorig = list(range(0, h))
                pixelsorig = origGreen[xxorig, int(w / 2)] / 1
                pixelsDPP0d = DPP0dGreen[xx, int(width / 2)] / 1
                pixelsDPP100d = DPP100dGreen[xx, int(width / 2)] / 1
                pixelsVdpp = Vdpp[xx, int(width / 2)] * 300
            else :
                pixelsorig = origGreen[xy[:, 0], xy[:, 1]] / 1
                pixelsDPP0d = DPP0dGreen[xy[:, 0], xy[:, 1]] / 1
                pixelsDPP100d = DPP100dGreen[xy[:, 0], xy[:, 1]] / 1
                pixelsVdpp = Vdpp[xy[:, 0], xy[:, 1]] * 66000

        if method == "Lab":
            if cross == "vertical":
                pixelsorig = origL[xx, int(width / 2)] / 1
                pixelsDPP0d = DPP0dL[xx, int(width / 2)] / 1
                pixelsDPP100d = DPP100dL[xx, int(width / 2)] / 1
                pixelsVdpp = Vdpp[xx, int(width / 2)] * 300
            else :
                pixelsorig = origL[xy[:, 0], xy[:, 1]] / 1
                pixelsDPP0d = DPP0dGreen[xy[:, 0], xy[:, 1]] / 1
                pixelsDPP100d = DPP100dGreen[xy[:, 0], xy[:, 1]] / 1
                pixelsVdpp = Vdpp[xy[:, 0], xy[:, 1]] * 66000


        # plt.clf()
        if imgNo == 1 :
            plt.plot(pixelsorig, 'r--')
            # plt.plot(pixelsDPP0d, 'r:')
            # plt.plot(pixelsDPP100d, 'r')
            # plt.plot(pixelsVdpp, 'r')
        if imgNo == 2 :
            plt.plot(pixelsorig, 'g--')
            plt.plot(pixelsDPP0d, 'g:')
            plt.plot(pixelsDPP100d, 'g')
            # plt.plot(pixelsVdpp, 'g')
        if imgNo == 3 :
            plt.plot(pixelsorig, 'b--')
            plt.plot(pixelsDPP0d, 'b:')
            plt.plot(pixelsDPP100d, 'b')
            # plt.plot(pixelsVdpp, 'b')
        if imgNo == 4 :
            plt.plot(pixelsorig, 'y--')
            plt.plot(pixelsDPP0d, 'y:')
            plt.plot(pixelsDPP100d, 'y')
            # plt.plot(pixelsVdpp, 'y')
        if imgNo == 5 :
            plt.plot(pixelsorig, 'c--')
            plt.plot(pixelsDPP0d, 'c:')
            plt.plot(pixelsDPP100d, 'c')
            # plt.plot(pixelsVdpp, 'c')
        if imgNo == 6 :
            plt.plot(pixelsorig, 'k--')
            plt.plot(pixelsDPP0d, 'k:')
            plt.plot(pixelsDPP100d, 'k')
            # plt.plot(pixelsVdpp, 'k')

        plt.savefig("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/%s/cross%s_%.2d.TIF" % (set, info, imgNo), bbox_inches='tight')

    else :
        # plot in 3d
        Xdpp, Ydpp, Zdpp = build3Dmesh(Vdpp)
        Xps, Yps, Zps = build3Dmesh(Vps)
        fig = plt.figure(figsize=(14,6))

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        p = ax.plot_surface(Xdpp, Ydpp, Zdpp, rstride=1, cstride=1, cmap="coolwarm", linewidth=0, antialiased=True)

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        p = ax.plot_surface(Xps, Yps, Zps, rstride=1, cstride=1, cmap="coolwarm", linewidth=0, antialiased=True)

        plt.savefig("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/%s/plot%s_%.2d.TIF" % (set, info, imgNo), bbox_inches='tight')

    # verify that PS applies the same vign image to all 3 RGB channels
    if method == "Blue":
        DPP100dBlue = np.multiply(Vdpp, np.asarray(DPP0dBlue, dtype=np.floating))
        DPP100dGreen = np.multiply(Vdpp, np.asarray(DPP0dGreen, dtype=np.floating))
        DPP100dRed = np.multiply(Vdpp, np.asarray(DPP0dRed, dtype=np.floating))

        PS100dBlue = np.multiply(Vps, np.asarray(PS0dBlue, dtype=np.floating))
        PS100dGreen = np.multiply(Vps, np.asarray(PS0dGreen, dtype=np.floating))
        PS100dRed = np.multiply(Vps, np.asarray(PS0dRed, dtype=np.floating))

        #put it back
        DPPdevegn = DPP0d.copy()
        DPPdevegn[:, :, 0] = DPP100dBlue.astype('int')
        DPPdevegn[:, :, 1] = DPP100dGreen.astype('int')
        DPPdevegn[:, :, 2] = DPP100dRed.astype('int')

        PSdevegn = PS0d.copy()
        PSdevegn[:, :, 0] = PS100dBlue.astype('int')
        PSdevegn[:, :, 1] = PS100dGreen.astype('int')
        PSdevegn[:, :, 2] = PS100dRed.astype('int')

    elif method == "Lab":
        DPP100dL = np.multiply(Vdpp, np.asarray(DPP0dL, dtype=np.floating))
        PS100dL = np.multiply(Vps, np.asarray(PS0dL, dtype=np.floating))

        #put it back
        DPPdevegn = DPP0dLab.copy()
        DPPdevegn[:, :, 0] = DPP100dL.astype('int')
        PSdevegn = PS0dLab.copy()
        PSdevegn[:, :, 0] = PS100dL.astype('int')

        # back to BGR
        DPPdevegn = cv2.cvtColor(DPPdevegn, cv2.COLOR_Lab2BGR)
        PSdevegn = cv2.cvtColor(PSdevegn, cv2.COLOR_Lab2BGR)

    DPPdevegn[xy[:, 0], xy[:, 1]] = [0, 0, 255]
    cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/%s/DPPImgOrigDevegn%s_%.2d.TIF" % (set, info, imgNo) , DPPdevegn)

    # the error
    err = abs(DPPdevegn.astype('float') - DPP100d.astype('float'))
    err = (err > 1 ).astype('int') # get ride of tiny errors due to rounding out
    mx = err.max()
    errNormalized = cv2.normalize(err, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # normalize it to 0-255 for better seeing tiny errores
    cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/%s/errNormalized%s_%.2d.TIF" % (set, info, imgNo), errNormalized)

v1 = VdppList[0]
v2 = VdppList[1]
v12 = v2 - v1
mx1 = v12.max()
mi1 = v12.min()

v1 = VdppList[0]
v2 = VdppList[2]
v12 = v2 - v1
mx1 = v12.max()
mi1 = v12.min()

v1 = VdppList[0]
v2 = VdppList[3]
v12 = v2 - v1
mx1 = v12.max()
mi1 = v12.min()

cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/%s/Z%.2d.TIF" % (set,12), v12)
v12Normalized = cv2.normalize(v12, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # normalize it to 0-255 for better seeing tiny errores
cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/%s/ZNormalized%.2d.TIF" % (set,12), v12Normalized)