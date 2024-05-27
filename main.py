import cv2

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import xml.etree.ElementTree as ET
tree = ET.parse('Canon EOS-1Ds Mark III (Canon EF 24-105mm f3.5-5.6 IS STM) - RAW.xml')
root = tree.getroot()

for c1 in root:
    for c2 in c1:
        for c3 in c2:
            for c4 in c3:
                for c5 in c4:
                    for c6 in c5:
                        print(c6.tag, c6.attrib)

VignetteModels =[]
for VignetteModel in root.iter('{stcamera}VignetteModel'):
    FocalLengthX = VignetteModel.get('{stcamera}FocalLengthX')
    FocalLengthY = VignetteModel.get('{stcamera}FocalLengthY')
    VignetteModelParam1 = VignetteModel.get('{stcamera}VignetteModelParam1')
    VignetteModelParam2 = VignetteModel.get('{stcamera}VignetteModelParam2')
    VignetteModelParam3 = VignetteModel.get('{stcamera}VignetteModelParam3')
    print('FocalLengthX =', FocalLengthX)
    print('FocalLengthY =', FocalLengthY)
    print('VignetteModelParam1 =', VignetteModelParam1)
    print('VignetteModelParam2 =', VignetteModelParam2)
    print('VignetteModelParam3 =', VignetteModelParam3)
    VignetteModels.append([float(FocalLengthX), float(FocalLengthY), float(VignetteModelParam1), float(VignetteModelParam2), float(VignetteModelParam3)])
    parent_map = {c: p for p in root.iter() for c in p}
    sa = parent_map.get('{stCamera}Author')

VignetteModels =[]
for li in root.iter('{ }li'):
    des = li.find('{ }Description')
    if des is not None : #skip AlternateLensNames
        PerspectiveModel = des.find('{stcamera}PerspectiveModel')
        if PerspectiveModel is not None:  # skip AlternateLensNames
            for PM_el in PerspectiveModel:
                VignetteModel = PerspectiveModel[0].find('{stcamera}VignetteModel')
                if VignetteModel is not None:
                    FocalLengthX = VignetteModel.get('{stcamera}FocalLengthX')
                    FocalLengthY = VignetteModel.get('{stcamera}FocalLengthY')
                    VignetteModelParam1 = VignetteModel.get('{stcamera}VignetteModelParam1')
                    VignetteModelParam2 = VignetteModel.get('{stcamera}VignetteModelParam2')
                    VignetteModelParam3 = VignetteModel.get('{stcamera}VignetteModelParam3')
                    FocalLength = des.get('{stcamera}FocalLength')
                    FocusDistance = des.get('{stcamera}FocusDistance')
                    ApertureValue = des.get('{stcamera}ApertureValue')
                    print('FocalLengthX =', FocalLengthX)
                    print('FocalLengthY =', FocalLengthY)
                    print('VignetteModelParam1 =', VignetteModelParam1)
                    print('VignetteModelParam2 =', VignetteModelParam2)
                    print('VignetteModelParam3 =', VignetteModelParam3)
                    print('FocalLength =', FocalLength)
                    print('FocusDistance =', FocusDistance)
                    print('ApertureValue =', ApertureValue)
                    VignetteModels.append([float(FocalLengthX), float(FocalLengthY), float(VignetteModelParam1), float(VignetteModelParam2), float(VignetteModelParam3),
                                           float(FocalLength), float(FocusDistance), float(ApertureValue)])

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

def generateDevignettingImg(a, FocalLengthX, FocalLengthY, rows, cols) :
    ImageXCenter = 0.5
    ImageYCenter = 0.5
    Dmax = max(rows, cols)  # 5640x3752
    # Dmax = np.sqrt(rows*rows+cols*cols)

    scale = 1.0
    Dmax = Dmax * scale
    u0 = ImageXCenter * Dmax
    v0 = ImageYCenter * Dmax
    fx = FocalLengthX * Dmax
    fy = FocalLengthY * Dmax
    devignette_factor = np.ones((int(rows / 2), int(cols / 2)), dtype=np.float32)

    vign_param = [0, 0, 0, 0]
    param0Sqr = a[0] * a[0]
    vign_param[0] = - a[0]
    vign_param[1] = param0Sqr + a[1]
    vign_param[2] = param0Sqr * a[0] - 2.0 * a[0] * a[1] + a[2]
    vign_param[3] = param0Sqr * param0Sqr + a[1] * a[1] + 2.0 * a[0] * a[2] - 3.0 * param0Sqr * a[1]

    for u in range(0, int(rows / 2)):  # TODO: make sure it is even
        for v in range(0, min(u + 1, int(cols / 2))):
            x = (u) / fx
            y = (v) / fy
            rsqr = x * x + y * y
            devignette_factor[u, v] = 1 + rsqr * (vign_param[0] + rsqr * ((vign_param[1]) - (vign_param[2]) * rsqr + (vign_param[3]) * rsqr * rsqr))
            if u < int(cols / 2):
                devignette_factor[v, u] = devignette_factor[u, v]

    devignette_factor = np.hstack((np.fliplr(devignette_factor), devignette_factor))
    devignette_factor = np.vstack((np.flipud(devignette_factor), devignette_factor))
    return devignette_factor

cross = "vertical"
set = "synthetic"
set = "test"
set = "set3"
set = "test6"

info = ""

F24FDp39AV4 = [-0.461002, -0.551963, 0.218801]
F24FDp39AV4FocalLengthX = 0.618716
F24FDp39AV4FocalLengthY = 0.618716

F24FDp39AV4970854 = [-0.465353, 0.421187, -0.671476]
F24FDp39AV4970854FocalLengthX = 0.618716
F24FDp39AV4970854FocalLengthY = 0.618716

F24FDp39AV4 = [-0.461002, -0.551963, 0.218801]
F24FDp39AV4FocalLengthX = 0.618716
F24FDp39AV4FocalLengthY = 0.618716

F24FDp79AV4970854 = [-0.478265, 0.427661, -0.731867]
F24FDp79AV4970854FocalLengthX = 0.637037
F24FDp79AV4970854FocalLengthY = 0.637037
F24FD10000AV4970854 = [-0.51399, 0.489486, -0.883353]
F24FD10000AV4970854FocalLengthX = 0.662776
F24FD10000AV4970854FocalLengthY = 0.662776

F35FDp39AV4970854 = [-0.911247, 1.723291, -3.338135]
F35FDp39AV4970854FocalLengthX = 0.935852
F35FDp39AV4970854FocalLengthY = 0.935852
F35FD10000AV4970854 = [-0.897462, 1.495122, -2.974976]
F35FD10000AV4970854FocalLengthX = 0.952012
F35FD10000AV4970854FocalLengthY = 0.952012

# according to PS lens correction filter
F24FDp79AV4 = [-0.589773, -0.405988, 0.169278]
F24FDp79AV4FocalLengthX = 0.637037
F24FDp79AV4FocalLengthY = 0.637037
F24FD10000AV4 = [-0.715117, -0.28464, 0.121978]
F24FD10000AV4FocalLengthX = 0.662776
F24FD10000AV4FocalLengthY = 0.662776

# gerwin blind shot!
F70FDp39AV4970854 = [-4.008772, 10.321273, -7.723757]
F70FDp39AV4970854FocalLengthX = 1.755752
F70FDp39AV4970854FocalLengthY = 1.755752

if(1):
    A = F24FDp39AV4
    FocalLengthX1 = F24FDp39AV4FocalLengthX
    FocalLengthY1 = F24FDp39AV4FocalLengthY
    B = F24FDp39AV4
    FocalLengthX2 = F24FDp39AV4FocalLengthX
    FocalLengthY2 = F24FDp39AV4FocalLengthY
if(0):
    A = F24FDp79AV4
    FocalLengthX1 = F24FDp79AV4FocalLengthX
    FocalLengthY1 = F24FDp79AV4FocalLengthY
    B = F24FDp79AV4
    FocalLengthX2 = F24FDp79AV4FocalLengthX
    FocalLengthY2 = F24FDp79AV4FocalLengthY
if(0):
    A = F24FD10000AV4
    FocalLengthX1 = F24FD10000AV4FocalLengthX
    FocalLengthY1 = F24FD10000AV4FocalLengthY
    B = F24FD10000AV4
    FocalLengthX2 = F24FD10000AV4FocalLengthX
    FocalLengthY2 = F24FD10000AV4FocalLengthY

if(0):
    A = F70FDp39AV4970854
    FocalLengthX1 = F70FDp39AV4970854FocalLengthX
    FocalLengthY1 = F70FDp39AV4970854FocalLengthY
    B = F70FDp39AV4970854
    FocalLengthX2 = F70FDp39AV4970854FocalLengthX
    FocalLengthY2 = F70FDp39AV4970854FocalLengthY

facA = (np.log(10000) - np.log(4.14))/(np.log(10000)-np.log(0.79))
facA = 0#(10000 - 4.14)/(10000-0.79)

facB = 1.0 - facA
FocalLengthX = FocalLengthX1*facA + FocalLengthX2* facB
FocalLengthY = FocalLengthY1*facA + FocalLengthY2* facB
a = [0, 0, 0]
a[0] = A[0] * facA + B[0] * facB
a[1] = A[1] * facA + B[1] * facB
a[2] = A[2] * facA + B[2] * facB

a1 = a[0]
a2 = a[1]
a3 = a[2]

img0d = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/%s/0d%.2d.tif" % (set, 1),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
height, width,_ = img0d.shape
devignette_factor = generateDevignettingImg(a, FocalLengthX, FocalLengthY, height, width)

xx = list(range(0, height))  # numpy.arange(height/2, height, 2)
yy = list(range(0, width))
xy = getCirclePixels(int(height / 2), int(width / 2), int(width / 2.3), height, width)

if cross == "vertical":
    pixelsdevignette_factor = devignette_factor[xx, int(width / 2)] * 66000
    pixelsdevignette_factor = np.clip(pixelsdevignette_factor, 0, 120000)
else:
    pixelsdevignette_factor = devignette_factor[xy[:, 0], xy[:, 1]] * 66000
    pixelsdevignette_factor = np.clip(pixelsdevignette_factor, 0, 120000)

plt.clf()
# title = "FL:%.2f, FD:%.3f, AS:%.4f" % (VignetteModel[5], VignetteModel[6], VignetteModel[7])
# plt.title(title)
plt.plot(pixelsdevignette_factor, 'k')

for imgNo in range(2,5) :
    img0d = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/%s/0d%.2d.tif" % (set, imgNo), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img100d = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/%s/100d%.2d.tif" % (set, imgNo), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    img0dBlue = img0d[:, :, 0].astype('float').copy()
    img0dGreen = img0d[:, :, 1].astype('float').copy()
    img0dRed = img0d[:, :, 2].astype('float').copy()
    img100dBlue = img100d[:, :, 0].astype('float').copy()
    img100dGreen = img100d[:, :, 1].astype('float').copy()
    img100dRed = img100d[:, :, 2].astype('float').copy()

    Vimg = np.asarray(img100dGreen, dtype=np.float) / (np.asarray(img0dGreen, dtype=np.float) + 0)

    VNormalized = cv2.normalize(Vimg, None, 0, 255, cv2.NORM_MINMAX)  # Convert to normalized floating point
    # VNormalized = Vimg * 100
    cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/VNormalized%s_%.2d.TIF" % (info, imgNo), Vimg)

    if cross == "vertical" :
        pixelsimg0d = img0dGreen[xx, int(width / 2)] / 1
        pixelsimg100d = img100dGreen[xx, int(width / 2)] / 1
        pixelsVimg = Vimg[xx, int(width / 2)] * 66000
        pixelsVimg = np.clip(pixelsVimg, 0, 120000)

    else :
        pixelsimg0d = img0dGreen[xy[:, 0], xy[:, 1]] / 1
        pixelsimg100d = img100dGreen[xy[:, 0], xy[:, 1]] / 1
        pixelsVimg = Vimg[xy[:, 0], xy[:, 1]] * 66000
        pixelsVimg = np.clip(pixelsVimg, 0, 120000)

    # apply the same vign image to all 3 RGB channels
    img100dBlue = np.multiply(devignette_factor, np.asarray(img0dBlue, dtype=np.floating))
    img100dGreen = np.multiply(devignette_factor, np.asarray(img0dGreen, dtype=np.floating))
    img100dRed = np.multiply(devignette_factor, np.asarray(img0dRed, dtype=np.floating))

    # put it back
    imgdevegn = img0d.copy().astype('float')
    imgdevegn[:, :, 0] = img100dBlue
    imgdevegn[:, :, 1] = img100dGreen
    imgdevegn[:, :, 2] = img100dRed

    mx = np.max(imgdevegn)
    imgdevegn = np.clip(imgdevegn, 0, 65535)
    mx = np.max(imgdevegn)
    imgdevegn = imgdevegn.astype('uint16')
    mx = np.max(imgdevegn)

    pixelsimgDeve100d = imgdevegn[:, :, 1][xx, int(width / 2)]

    if imgNo == 1 :
        plt.plot(pixelsimg0d, 'r--')
        plt.plot(pixelsimg100d, 'r')
        plt.plot(pixelsVimg, 'r')
        plt.plot(pixelsimgDeve100d, 'r:')
    if imgNo == 2 :
        plt.plot(pixelsimg0d, 'g--')
        plt.plot(pixelsimg100d, 'g')
        plt.plot(pixelsVimg, 'g')
        plt.plot(pixelsimgDeve100d, 'g:')
    if imgNo == 3 :
        plt.plot(pixelsimg0d, 'b--')
        plt.plot(pixelsimg100d, 'b')
        plt.plot(pixelsVimg, 'b')
        plt.plot(pixelsimgDeve100d, 'b:')
    if imgNo == 4 :
        plt.plot(pixelsimg0d, 'y--')
        plt.plot(pixelsimg100d, 'y')
        plt.plot(pixelsVimg, 'y')
    if imgNo == 5 :
        plt.plot(pixelsimg0d, 'c--')
        plt.plot(pixelsimg100d, 'c')
        plt.plot(pixelsVimg, 'c')
    if imgNo == 6 :
        plt.plot(pixelsimg0d, 'k--')
        plt.plot(pixelsimg100d, 'k')
        plt.plot(pixelsVimg, 'k')

    plt.savefig("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/cross%s_%.2d.TIF" % ( info, imgNo), bbox_inches='tight')


    # imgdevegn[xy[:, 0], xy[:, 1]] = [0, 0, 65530]
    cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/100d%s%.2dmy.TIF" % ( info, imgNo) , imgdevegn)

    # the error
    err = abs(imgdevegn.astype('float') - img100d.astype('float'))
    # err = (err > 1 ).astype('int') # get ride of tiny errors due to rounding out
    mx = err.max()
    errNormalized = cv2.normalize(err, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # normalize it to 0-255 for better seeing tiny errores
    cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/errNormalized%s_%.2d.TIF" % (info, imgNo), errNormalized)
