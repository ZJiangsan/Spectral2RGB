# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:01:39 2019

@author: jizh
"""
import cv2
import h5py
import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.color import colorconv
import colour
from spectral import *
from PIL import Image
import csv


def spectral2XYZ_img_vectorized(cmfs, R):
    x_bar, y_bar, z_bar = colour.tsplit(cmfs)  #
    plt.close('all')
    plt.plot(np.array([z_bar, y_bar, x_bar]).transpose())
    plt.savefig('cmf_cie1964_10.png')
    plt.close('all')
    # illuminant. 
    S = colour.ILLUMINANTS_RELATIVE_SPDS['E'].values[0:31] / 100.
    dw = 10
    k = 100 / (np.sum(y_bar * S) * dw)

    X_p = R * x_bar * S * dw
    Y_p = R * y_bar * S * dw
    Z_p = R * z_bar * S * dw

    XYZ = k * np.sum(np.array([X_p, Y_p, Z_p]), axis=-1)
    XYZ = np.rollaxis(XYZ, 1, 0)
    return XYZ

def spectral2XYZ_img(hs, cmf_name):

    h, w, c = hs.shape
    hs = hs.reshape(-1, c)
    cmfs = get_cmfs(cmf_name=cmf_name, nm_range=(400., 701.), nm_step=10, split=False)
    XYZ = spectral2XYZ_img_vectorized(cmfs, hs)  # (nb_px, 3)
    XYZ = XYZ.reshape((h, w, 3))
    return XYZ


def spectral2sRGB_img(spectral, cmf_name, image_data_format='channels_last'):
    XYZ = spectral2XYZ_img(hs=spectral, cmf_name=cmf_name, image_data_format=image_data_format)
    sRGB = colorconv.xyz2rgb(XYZ/100.)
    return sRGB

def get_cmfs(cmf_name='cie1964_10', nm_range=(400., 701.), nm_step=10, split=True):
    if cmf_name == 'cie1931_2':
        cmf_full_name = 'CIE 1931 2 Degree Standard Observer'
    elif cmf_name == 'cie2012_2':
        cmf_full_name = 'CIE 2012 2 Degree Standard Observer'
    elif cmf_name == 'cie2012_10':
        cmf_full_name = 'CIE 2012 10 Degree Standard Observer'
    elif cmf_name == 'cie1964_10':
        cmf_full_name = 'CIE 1964 10 Degree Standard Observer'
    else:
        raise AttributeError('Wrong cmf name')
    cmfs = colour.STANDARD_OBSERVERS_CMFS[cmf_full_name]

    # subsample and trim range
    ix_wl_first = np.where(cmfs.wavelengths == nm_range[0])[0][0]
    ix_wl_last = np.where(cmfs.wavelengths == nm_range[1]+1.)[0][0]
    cmfs = cmfs.values[ix_wl_first:ix_wl_last:int(nm_step), :]

    if split:
        x_bar, y_bar, z_bar = colour.tsplit(cmfs)
        return x_bar, y_bar, z_bar
    else:
        return cmfs



def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


## read the wave bands of hyperspectral image from specim IQ


with open('wavelength_interval.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    wi = []
    i = 0
    for row in readCSV:
        print(i)
        print(row)
        if i >0:
            wi.append(round(float(row[0]),3))
        i +=1



# the wavebands of hyperspectral image locate in visual range
visual_spec = list(range(400, 701, 10))
#wi = np.array(wi)

x_cor = []
for i in visual_spec:
    x_cor_i = np.where(abs(wi - i) == min(abs(wi - i)))
    x_cor.append(x_cor_i[0].tolist()[0])

wi_spec = wi[x_cor]

x_shit = get_cmfs(cmf_name = "cie1931_2", nm_range=(400., 700.), nm_step=10, split=True)

## eg
mat =  h5py.File("input_hsi.h5",'r')
hyper = mat['img']
hyper.shape

hyper= np.transpose(hyper, [2,1,0])
spectral_x = hyper[:,:,x_cor] ## extract the wavebands in vidual range
shi_rgb =spectral2sRGB_img(spectral_x, "cie1931_2", image_data_format='channels_last')

shi_rgb.shape

plt.figure(figsize = (6,6))
plt.hist(shi_rgb.reshape((512*512*3)), bins = 50)
plt.show()

# stretch the spectrum based on the reflectance histogram
shi_rgb_x_0 = (shi_rgb-0.21)/0.8
shi_rgb_x_0[shi_rgb_x_0>1]=1
shi_rgb_x_0[shi_rgb_x_0<0] = 0

##   costumize the values in each channel to enchance or reduce its prominance
#shi_rgb_x_0[:,:,2] = shi_rgb_x_0[:,:,2]*1.1
#shi_rgb_x_0[:,:,1] = shi_rgb_x_0[:,:,1]*1.21 # enhance green if the green color is not well presented
shi_rgb_x_0[:,:,0] = shi_rgb_x_0[:,:,0]*0.85 ## reduce the red as the image looks a little red

shi_rgb_x_0[shi_rgb_x_0>1]=1

shi_rgb_x = np.array(shi_rgb_x_0*255).astype(np.uint8)
save_image(shi_rgb_x, "output_rgb.png")



## do gamma correction to adjust the brightness if necessary
shi_rgb_x = np.array(shi_rgb*255).astype(np.uint8)
shi_rgb_x = cv2.cvtColor(shi_rgb_x, cv2.COLOR_BGR2RGB)

gamma = 1.0
table = np.array([((i / 255.0) ** (1/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")

shi_rgb_x_gc= cv2.LUT(shi_rgb_x, table)
cv2.imshow("Gamma corrected", shi_rgb_x_gc)




