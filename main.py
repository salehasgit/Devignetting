import numpy as np
import math
import cv2

def getCirclePixels(x0, y0, R, imH, imW):
    # sample points
    theta = np.linspace(0.1, 2 * np.pi-0.1, 2048)  # make it finer for finer circles
    # the pixels that get hit
    xy = [xy for xy in zip( ( - R * np.sin(theta) + x0).astype(int), (R * np.cos(theta) + y0).astype(int) ) if xy[0] >= 0 and xy[0] < imH and xy[1] >= 0 and xy[1] < imW]
    return np.array(xy)

PS0d = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/test4/PS0d.jpg", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
PS100d = cv2.imread("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/test4/PS100d.jpg", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# PS0d = PS0d / 65535 *255
# PS100d = PS100d / 65535 *255
PS00dMAX = PS100d.max()

rows, cols, _ = PS0d.shape
xy = getCirclePixels(int(rows / 2), int(cols / 2), int(cols / 2.3), rows, cols)

F24FD1000AV6 = [-0.563363, 0.699109, -0.898485]
F24FD1000AV49 = [-0.51399, 0.489486, -0.883353]
interpol = [(F24FD1000AV6[0]+F24FD1000AV49[0])/2, (F24FD1000AV6[1]+F24FD1000AV49[1])/2,(F24FD1000AV6[2]+F24FD1000AV49[2])/2]

F24FD1000AV4 = [-0.715117, -0.28464, 0.121978]

ImageXCenter = 0.490518
ImageYCenter = 0.4873
FocalLengthX = 0.662776
FocalLengthY = 0.662776

a = F24FD1000AV4
a1 = a[0]
a2 = a[1]
a3 = a[2]

Dmax = max(rows, cols) #5640
u0 = ImageXCenter * Dmax
v0 = ImageYCenter * Dmax
fx = FocalLengthX * Dmax
fy = FocalLengthY * Dmax

ps = 36/5616   #36x24
ps2 = 36/5472
# fx = 24/ps2
# fy = fx

vignette_factor = np.ones((int(rows/2), int(cols/2)), dtype=np.float32)

for u in range(0, int(rows/2)): #TODO: make sure it is even
    for v in range(0, min(u+1, int(cols/2)) ):
        x = (u ) / fx
        y = (v ) / fy
        r = pow(pow(x, 2) + pow(y, 2), 0.5 )
        vignette_factor[u, v] = 1 - a1 * pow(r, 2) + (pow(a1, 2) - a2) * pow(r, 4) - (pow(a1, 3) - 2 * a1 * a2 + a3) * pow(r, 6) + (pow(a1, 4) + pow(a2, 2) + 2 * a1 * a3 - 3 * pow(a1, 2) * a2 ) * pow(r, 8)
        if u < int(cols / 2):
            vignette_factor[v, u] = 1 - a1 * pow(r, 2) + (pow(a1, 2) - a2) * pow(r, 4) - (pow(a1, 3) - 2 * a1 * a2 + a3) * pow(r, 6) + (pow(a1, 4) + pow(a2, 2) + 2 * a1 * a3 - 3 * pow(a1, 2) * a2 ) * pow(r, 8)

vignettingImg = np.hstack((np.fliplr(vignette_factor), vignette_factor))
vignettingImg = np.vstack((np.flipud(vignettingImg), vignettingImg))
VNormalized = cv2.normalize(vignettingImg, None, 0, 255, cv2.NORM_MINMAX)  # Convert to normalized floating point
cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/V.jpg", VNormalized)

vigAve = vignettingImg[xy[:, 0], xy[:, 1]]
vigAve = np.average(vigAve)
print(vigAve)
vigAvePS = (PS0d/PS100d)[:, :, 1]
vigAvePS = vigAvePS[xy[:, 0], xy[:, 1]]
vigAvePS = np.average(1/vigAvePS)
print(vigAvePS)

B, G, R = cv2.split(PS0d)
B = B * vignettingImg
G = G * vignettingImg
R = R * vignettingImg
correctedImg = cv2.merge((B,G,R))
# correctedImg = cv2.normalize(correctedImg, None, 0, PS00dMAX, cv2.NORM_MINMAX)  # Convert to normalized floating point

cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/PS0dpy.jpg", PS0d)
cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/PS100dpy.jpg", PS100d)
cv2.imwrite("/Users/sm/Dropbox (VR Holding BV)/de-vignetting/res/My100d.jpg", (correctedImg))