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
