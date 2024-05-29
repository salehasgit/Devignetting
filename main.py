import cv2

import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import xml.etree.ElementTree as ET

tree = ET.parse('Canon EOS-1Ds Mark III (Canon EF 24-105mm f3.5-5.6 IS STM) - RAW.xml')
root = tree.getroot()

# for c1 in root:
#     for c2 in c1:
#         for c3 in c2:
#             for c4 in c3:
#                 for c5 in c4:
#                     for c6 in c5:
#                         print(c6.tag, c6.attrib)

VignetteModels = []
for li in root.iter('{ }li'):
    des = li.find('{ }Description')
    if des is not None:  # skip AlternateLensNames
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
                    ResidualMeanError = des.get('{stcamera}ResidualMeanError')
                    ResidualStandardDeviation = des.get('{stcamera}ResidualStandardDeviation')
                    print('FocalLengthX =', FocalLengthX)
                    print('FocalLengthY =', FocalLengthY)
                    print('VignetteModelParam1 =', VignetteModelParam1)
                    print('VignetteModelParam2 =', VignetteModelParam2)
                    print('VignetteModelParam3 =', VignetteModelParam3)
                    print('FocalLength =', FocalLength)
                    print('FocusDistance =', FocusDistance)
                    print('ApertureValue =', ApertureValue)
                    print('ResidualMeanError =', ResidualMeanError)
                    print('ResidualStandardDeviation =', ResidualStandardDeviation)
                    VignetteModels.append([float(FocalLengthX), float(FocalLengthY), float(VignetteModelParam1),
                                           float(VignetteModelParam2), float(VignetteModelParam3),
                                           float(FocalLength), float(FocusDistance), float(ApertureValue),
                                           0.0 if ResidualMeanError==None else float(ResidualMeanError), 0.0 if ResidualStandardDeviation==None else float(ResidualStandardDeviation)])

def calculParams(focalLength,focusDist, aperture):

    euler = np.exp(1.0)

    # find the frames with the least distance, focal length wise
    pLow = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pHigh = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    focalLengthLog = np.log(focalLength)
    focusDistLog = 0
    if focusDist > 0 :
        focusDistLog = np.log(focusDist) + euler

    # Pass 1: determining best focal length, if possible different focusDistances (for the focDist is not given case)
    for VignetteModel in VignetteModels:
        f = VignetteModel[5]
        if f <= focalLength & ( all(v == 0 for v in pLow)  | f > pLow[5] | (focusDist == 0 & f == pLow[5] & pLow[6] > VignetteModel[6] ) ) :
            pLow = VignetteModel
        if f >= focalLength & ( all(v == 0 for v in pHigh) | f < pLow[5] | (focusDist == 0 & f == pLow[5] & pLow[6] < VignetteModel[6])):
            pHigh = VignetteModel


    if all(v == 0 for v in pLow):
        pLow = pHigh
    elif all(v == 0 for v in pHigh):
        pHigh = pLow

    else :
         # Pass 2: We have some, so take the best aperture for vignette and best focus for CA and distortion
         #  there are usually several frame per focal length. In the end pLow will have both flen and apterure/focdis below the target,
         #  and vice versa pHigh
        bestFocLenLow = pLow[5]
        bestFocLenHigh = pHigh[5]

        for VignetteModel in VignetteModels:
            aper = VignetteModel[7]
            focDist = VignetteModel[6]
            focDistLog = np.log(focDist) + euler

            meanErr = VignetteModel[8]
            lowMeanErr = pLow[8]
            highMeanErr = pHigh[8]


            if aperture > 0 : #the best aperture for vignette and distortion!?
                if (VignetteModel[5] == bestFocLenLow & ((aper == aperture & lowMeanErr > meanErr) | (aper >= aperture & aper < pLow[7] & pLow[7] > aperture) | (aper <= aperture & (pLow[7] > aperture | abs(aperture - aper) < abs(aperture - pLow[7]))))):
                        pLow = VignetteModel
                if (VignetteModel[5] == bestFocLenHigh & ((aper == aperture & highMeanErr > meanErr) | (aper <= aperture & aper > pHigh[7] & pHigh[7] < aperture) | (aper >= aperture & ( pHigh[7] < aperture | abs(aperture - aper) < abs(aperture - pHigh[7]))))):
                    pHigh = VignetteModel
                else:
                    # no focus distance available, just error
                    if VignetteModel[5] == bestFocLenLow & lowMeanErr > meanErr:
                        pLow = VignetteModel
                    if VignetteModel[5] == bestFocLenHigh & highMeanErr > meanErr:
                        pHigh = VignetteModel

    if not (all(v == 0 for v in pLow) &  all(v == 0 for v in pHigh)) :
        # average out the factors, linear interpolation in logarithmic scale
        facLow = 0.5
        focLenOnSpot = False # pretty often, since max/min are often as frames in LCP

        #There is as foclen range, take that as basis
        if pLow[5] < pHigh[5]:
            facLow = (np.log(pHigh[5]) - focalLengthLog) / (np.log(pHigh[5]) - np.log(pLow[5]))
        else:
            focLenOnSpot = pLow[5] == pHigh[5] & pLow[5] == focalLength

        # and average the other factor if available
        if pLow[7] < aperture & pHigh[7] > aperture:
             # Mix in aperture
            facAperLow = (pHigh[7] - aperture) / (pHigh[7] - pLow[7]);
            facLow = facAperLow if focLenOnSpot else 0.5 * facLow + 0.5 * facAperLow



        facA = facLow
        facB = 1.0 - facA

        foc_len_x = facA * pLow[0] + facB * pHigh[0]
        foc_len_y = facA * pLow[1] + facB * pHigh[1]
        # img_center_x = facA * a.img_center_x + facB * b.img_center_x;
        # img_center_y = facA * a.img_center_y + facB * b.img_center_y;
        # scale_factor = facA * a.scale_factor + facB * b.scale_factor;
        mean_error = facA * pLow[8] + facB * pHigh[8]

        param = [0, 0, 0]
        param[0] = facA * pLow[2] + facB * pHigh[2]
        param[1] = facA * pLow[3] + facB * pHigh[3]
        param[2] = facA * pLow[4] + facB * pHigh[4]

        return foc_len_x, foc_len_y, param


subsample = 20
def build3Dmesh(diff):
    h, w = diff.shape
    R = range(0, h, subsample)
    C = range(0, w, subsample)
    X = np.zeros((len(R), len(C)))
    Y = np.zeros((len(R), len(C)))
    Z = np.zeros((len(R), len(C)))
    for r in range(0, len(R)):
        for c in range(0, len(C)):
            X[r, c] = r
            Y[r, c] = c
            Z[r, c] = diff[R[r], C[c]]
    return X, Y, Z


def getCirclePixels(x0, y0, R, imH, imW):
    # sample points
    theta = np.linspace(0.1, 2 * np.pi - 0.1, 2048)  # make it finer for finer circles
    # the pixels that get hit
    xy = [xy for xy in zip((- R * np.sin(theta) + x0).astype(int), (R * np.cos(theta) + y0).astype(int)) if
          xy[0] >= 0 and xy[0] < imH and xy[1] >= 0 and xy[1] < imW]
    return np.array(xy)


def generateDevignettingImg(a, FocalLengthX, FocalLengthY, rows, cols):
    ImageXCenter = 0.5
    ImageYCenter = 0.5
    Dmax = max(rows, cols)  # 5640

    u0 = ImageXCenter * Dmax
    v0 = ImageYCenter * Dmax
    fx = FocalLengthX * Dmax
    fy = FocalLengthY * Dmax
    devignette_factor = np.ones((int(rows / 2), int(cols / 2)), dtype=np.float32)

    vign_param =[0, 0, 0, 0]
    param0Sqr = a[0] * a[0]
    vign_param[0] = - a[0]
    vign_param[1] = param0Sqr - a[1]
    vign_param[2] = param0Sqr * a[0] - 2.0 * a[0] * a[1] + a[2]
    vign_param[3] = param0Sqr * param0Sqr + a[1] * a[1] + 2.0 * a[0] * a[2] - 3.0 * param0Sqr * a[1]

    for u in range(0, int(rows / 2)):  # TODO: make sure it is even
        for v in range(0, min(u + 1, int(cols / 2))):
            x = (u) / fx
            y = (v) / fy
            rsqr = x*x + y*y
            devignette_factor[u, v] = 1 + rsqr * ( vign_param[0] + rsqr * ((vign_param[1]) - (vign_param[2]) * rsqr + (vign_param[3]) * rsqr * rsqr))

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

imgTmp = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/%s/0d%.2d.tif" % (set, 1), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
height, width, _ = imgTmp.shape
xx = list(range(0, height))  # numpy.arange(height/2, height, 2)
yy = list(range(0, width))
xy = getCirclePixels(int(height / 2), int(width / 2), int(width / 2.3), height, width)

counter = 0
for VignetteModel in VignetteModels:
    print('FL: ', int(VignetteModel[5]))
    if int(VignetteModel[5]) == 24 :
        counter = counter + 1
        print('counter: ', counter)

        a = VignetteModel[2:5]
        FocalLengthX = VignetteModel[0]
        FocalLengthY = VignetteModel[1]

        FocalLengthX, FocalLengthY, a  = calculParams(24, 4.14, 4)

        devignette_factor = generateDevignettingImg(a, FocalLengthX, FocalLengthY, height, width)
        VNormalized = cv2.normalize(devignette_factor, None, 0, 255, cv2.NORM_MINMAX)  # Convert to normalized floating point
        cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/V%.2d.jpg" % counter, VNormalized)

        if cross == "vertical":
            pixelsdevignette_factor = devignette_factor[xx, int(width / 2)] * 66000
            pixelsdevignette_factor = np.clip(pixelsdevignette_factor, 0, 120000)
        else:
            pixelsdevignette_factor = devignette_factor[xy[:, 0], xy[:, 1]] * 66000
            pixelsdevignette_factor = np.clip(pixelsdevignette_factor, 0, 120000)

        plt.clf()
        title = "FL:%.2f, FD:%.3f, AS:%.4f" % (VignetteModel[5], VignetteModel[6], VignetteModel[7])
        plt.title(title)
        plt.plot(pixelsdevignette_factor, 'k')

        for imgNo in range(2, 3):
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
            cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/VPS%s_%.2d.jpg" % (info, counter), VNormalized)

            if cross == "vertical":
                pixelsimg0d = img0dGreen[xx, int(width / 2)] / 1
                pixelsimg100d = img100dGreen[xx, int(width / 2)] / 1
                pixelsVimg = Vimg[xx, int(width / 2)] * 66000
                pixelsVimg = np.clip(pixelsVimg, 0, 100000)

            else:
                pixelsimg0d = img0dGreen[xy[:, 0], xy[:, 1]] / 1
                pixelsimg100d = img100dGreen[xy[:, 0], xy[:, 1]] / 1
                pixelsVimg = Vimg[xy[:, 0], xy[:, 1]] * 66000
                pixelsVimg = np.clip(pixelsVimg, 0, 100000)

            if imgNo == 1:
                plt.plot(pixelsimg0d, 'r--')
                plt.plot(pixelsimg100d, 'r')
                plt.plot(pixelsVimg, 'r')
            if imgNo == 2:
                plt.plot(pixelsimg0d, 'g--')
                plt.plot(pixelsimg100d, 'g')
                plt.plot(pixelsVimg, 'g')
            if imgNo == 3:
                plt.plot(pixelsimg0d, 'b--')
                plt.plot(pixelsimg100d, 'b')
                plt.plot(pixelsVimg, 'b')
            if imgNo == 4:
                plt.plot(pixelsimg0d, 'y--')
                plt.plot(pixelsimg100d, 'y')
                plt.plot(pixelsVimg, 'y')
            if imgNo == 5:
                plt.plot(pixelsimg0d, 'c--')
                plt.plot(pixelsimg100d, 'c')
                plt.plot(pixelsVimg, 'c')
            if imgNo == 6:
                plt.plot(pixelsimg0d, 'k--')
                plt.plot(pixelsimg100d, 'k')
                plt.plot(pixelsVimg, 'k')

            plt.savefig("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/cross%s_%.2d.TIF" % (info, counter), bbox_inches='tight')

            # verify that PS applies the same vign image to all 3 RGB channels
            img100dBlue = np.multiply(devignette_factor, np.asarray(img0dBlue, dtype=np.floating))
            img100dGreen = np.multiply(devignette_factor, np.asarray(img0dGreen, dtype=np.floating))
            img100dRed = np.multiply(devignette_factor, np.asarray(img0dRed, dtype=np.floating))

            # put it back
            imgdevegn = img0d.copy()
            imgdevegn[:, :, 0] = img100dBlue
            imgdevegn[:, :, 1] = img100dGreen
            imgdevegn[:, :, 2] = img100dRed

            imgdevegn = np.clip(imgdevegn, 0, np.iinfo('uint16').max)
            imgdevegn = imgdevegn.astype('uint16')
            imgdevegn[xy[:, 0], xy[:, 1]] = [0, 0, np.iinfo('uint16').max]
            cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/100d%s%.2dmy.TIF" % (info, counter), imgdevegn)

            # the error
            err = abs(imgdevegn.astype('float') - img100d.astype('float'))
            # err = (err > 1 ).astype('int') # get ride of tiny errors due to rounding out
            mx = err.max()
            errNormalized = cv2.normalize(err, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # normalize it to 0-255 for better seeing tiny errores
            cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/errNormalized%s_%.2d.TIF" % (info, counter), errNormalized)
