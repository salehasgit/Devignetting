# swift code for GPU vignetting https://stackoverflow.com/questions/9158629/how-to-apply-vignette-and-vintage-image-filter-in-app
import numpy as np
import math
import cv2

def getCirclePixels(x0, y0, R, imH, imW):
    # sample points
    theta = np.linspace(0.1, 2 * np.pi-0.1, 2048)  # make it finer for finer circles
    # the pixels that get hit
    xy = [xy for xy in zip( ( - R * np.sin(theta) + x0).astype(int), (R * np.cos(theta) + y0).astype(int) ) if xy[0] >= 0 and xy[0] < imH and xy[1] >= 0 and xy[1] < imW]
    return np.array(xy)

PS0d = cv2.imread("/de-vignetting/test4/0d03.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
PS100d = cv2.imread("/de-vignetting/test4/100d03.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# PS0d = PS0d / 65535 *255
# PS100d = PS100d / 65535 *255
PS100dMAX = PS100d.max()

rows, cols, _ = PS0d.shape
xy = getCirclePixels(int(rows / 2), int(cols / 2), int(cols / 2.3), rows, cols)

if(1) :
    F24FDp79AV4 = [-0.589773, -0.405988, 0.169278]
    FocalLengthX = 0.637037
    FocalLengthY = 0.637037

    # FocalLengthX = 0.952012
    # FocalLengthY = 0.952012

    a = F24FDp79AV4

if (0):
    F24FDp79AV4 = [-0.589773, -0.405988, 0.169278]
    FocalLengthX1 = 0.637037
    FocalLengthY1 = 0.637037
    F24FD1000AV4 = [-0.715117, -0.28464, 0.121978]
    FocalLengthX2 = 0.662776
    FocalLengthY2 = 0.662776

    a = [(F24FDp79AV4[0]+F24FD1000AV4[0])/2, (F24FDp79AV4[1]+F24FD1000AV4[1])/2,(F24FDp79AV4[2]+F24FD1000AV4[2])/2]
    FocalLengthX = (FocalLengthX1+FocalLengthX2)/2
    FocalLengthY = (FocalLengthY1+FocalLengthY2)/2

if(0):
    F105FD39AV6 = [-27.436744, 4454.30217, -394799.841178]
    FocalLengthX = 6.036768
    FocalLengthY = 6.036768
    a = F24FD1000AV4


a1 = a[0]
a2 = a[1]
a3 = a[2]

ImageXCenter = 0.5
ImageYCenter = 0.5
Dmax = max(rows, cols) #5640
u0 = ImageXCenter * Dmax
v0 = ImageYCenter * Dmax
fx = FocalLengthX * Dmax
fy = FocalLengthY * Dmax


fx = 4680
fy = 4680

ps = 36/5472
# fx = 24/ps
# fy = fx

vignette_factor = np.ones((int(rows/2), int(cols/2)), dtype=np.float32)
vignette_factor2 = np.ones((int(rows/2), int(cols/2)), dtype=np.float32)
for u in range(0, int(rows/2)): #TODO: make sure it is even
    for v in range(0, min(u+1, int(cols/2)) ):
        x = (u ) / fx
        y = (v ) / fy
        r = pow(pow(x, 2) + pow(y, 2), 0.5 )
        vignette_factor[u, v] = 1 - a1 * pow(r, 2) + (pow(a1, 2) - a2) * pow(r, 4) - (pow(a1, 3) - 2 * a1 * a2 + a3) * pow(r, 6) + (pow(a1, 4) + pow(a2, 2) + 2 * a1 * a3 - 3 * pow(a1, 2) * a2 ) * pow(r, 8)
        vignette_factor2[u, v] = 1 + a1 * pow(r, 2) + a2 * pow(r, 4)+ a3 * pow(r, 6)
        if u < int(cols / 2):
            vignette_factor[v, u] = 1 - a1 * pow(r, 2) + (pow(a1, 2) - a2) * pow(r, 4) - (pow(a1, 3) - 2 * a1 * a2 + a3) * pow(r, 6) + (pow(a1, 4) + pow(a2, 2) + 2 * a1 * a3 - 3 * pow(a1, 2) * a2 ) * pow(r, 8)
            vignette_factor2[v, u] = 1 + a1 * pow(r, 2) + a2 * pow(r, 4) + a3 * pow(r, 6)

vignettingImg = np.hstack((np.fliplr(vignette_factor), vignette_factor))
vignettingImg = np.vstack((np.flipud(vignettingImg), vignettingImg))
VNormalized = cv2.normalize(vignettingImg, None, 0, 255, cv2.NORM_MINMAX)  # Convert to normalized floating point
cv2.imwrite("/de-vignetting/res/V.jpg", VNormalized)


vignettingImg2 = np.hstack((np.fliplr(vignette_factor2), vignette_factor2))
vignettingImg2 = np.vstack((np.flipud(vignettingImg2), vignettingImg2))
VNormalized = cv2.normalize(vignettingImg2, None, 0, 255, cv2.NORM_MINMAX)  # Convert to normalized floating point
cv2.imwrite("/de-vignetting/res/V2.jpg", VNormalized)

vigAve = vignettingImg[xy[:, 0], xy[:, 1]]
vigAve = np.average(vigAve)
print(vigAve)
vigAvePS = (PS0d/PS100d)[:, :, 1]
vigAvePS = vigAvePS[xy[:, 0], xy[:, 1]]
vigAvePS = np.average(1/vigAvePS)
print(vigAvePS)
# vignettingImg = vignettingImg * vigAve / vigAvePS

B, G, R = cv2.split(PS0d)
B = B * vignettingImg
G = G * vignettingImg
R = R * vignettingImg
correctedImg = cv2.merge((B, G, R))

#deal with saturated pixels
#correctedImg = cv2.normalize(correctedImg, None, 0, PS100dMAX, cv2.NORM_MINMAX)  # Convert to normalized floating point
correctedImg = np.clip(correctedImg, 0, 65530)

# the error
err = abs(correctedImg.astype('float') - PS100d.astype('float'))
mx = err.max()
errNormalized = cv2.normalize(err, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # normalize it to 0-255 for better seeing tiny errores
cv2.imwrite("/de-vignetting/res/errNormalized.tif", errNormalized)

cv2.imwrite("/de-vignetting/res/0dpy.tif", PS0d)# after saving, is it identical to the orifginal file? yes
cv2.imwrite("/de-vignetting/res/100dpy.tif", PS100d)# after saving, is it identical to the orifginal file? yes
correctedImg = correctedImg.astype('uint16')
cv2.imwrite("/de-vignetting/res/100dmy.tif", (correctedImg))

