""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import cv2
import numpy as np
import pathlib
import re
from pyzbar import pyzbar

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def load_path(search_path, ext = '*.png'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key = natural_key)
    
    return all_files


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return np.float32(cv2.LUT(image.astype('uint8'), table))


def flat_field_correct(img=None):
    """ Performs a flat field correction in the image for reducing the intensity differences between the kernels due
        to lighting conditions.
        https://en.wikipedia.org/wiki/Flat-field_correction

    :param img: (optional) image to be processed, doesn't save the results in self.image. If not supplied, self.image
                will be processed.
    :return:
    """

    img_o = img.copy()

    num_dim = len(img_o.shape)
    if num_dim == 2:  # single channel img.
        img_o = np.expand_dims(img_o, 2)  # Add the channel dimension
    size = img_o.shape

    k_size = min(55, max(img_o.shape) // 3)  # kernel size for the median
    k_size = k_size + 1 if k_size % 2 == 0 else k_size  # Make sure it's an odd number

    n_channels = size[2]
    map = np.zeros(size)
    mean = np.zeros(n_channels)
    mat = np.zeros(size)
    for i in range(n_channels):
        map[:,:,i] = cv2.medianBlur(img_o[:,:,i], k_size)  # Estimate flat field image
        mean[i] = np.mean(img_o[:,:,i])
        mat[:,:,i] = (mean[i] / map[:,:,i])

        mat[:,:,i] = .05 + cv2.GaussianBlur(mat[:,:,i], (k_size,k_size), 1)
        img_o[:,:,i] = (img_o[:,:,i] *mat[:,:,i]).astype(np.float64)
        # img_o[:,:,i] = cv.equalizeHist(img_o[:,:,i])

    if num_dim == 2:  # single channel img
        img_o = np.squeeze(img_o, axis=2)  # Removes the channel dimension

    return img_o


def qr_detect(inputImage):
    inputImage = cv2.imread(inputImage, 0)
    h, w = inputImage.shape
    h0 = 0
    h1 = int(h//2) 
    w0 = int(w//4)
    w1 = int(3*w0)
        
    roi = inputImage[h0:h1, w0:w1]

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))    
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))

    gamma = 1.2
    blur_re = adjust_gamma(roi, gamma).astype('uint8')
    
    a = flat_field_correct(blur_re)
    th3 = 255 - cv2.threshold(a ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    cv2.erode(th3,kernel1, th3)    
    cv2.dilate(th3, kernel2, th3)
    cv2.dilate(th3, kernel2, th3)

    contours, hierarchy = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_sizes = [(cv2.contourArea(contour)) for contour in contours]
    contour_sizes = np.array(contour_sizes)
    
    index2 = np.argsort(contour_sizes)[::-1]
    index = contour_sizes[index2] > 4000
    
    for k in index2[index]:
        x, y, w, h = cv2.boundingRect(contours[k])
        
        if w < 100 or h < 100:
            pass
        else:
            enmask = roi[y:y+h, x:x+w].copy()
            
            cv2.medianBlur(enmask, 5, enmask)
            
            th3aux = 255 - cv2.threshold(enmask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            cv2.bitwise_and(enmask, th3aux, enmask)

            for i in range(0, 75):
                gamma = 1.0 + i*0.01
                adjust = adjust_gamma(enmask, gamma).astype('uint8')

                data = pyzbar.decode(adjust)
                
                if len(data) > 0:
                    break
            
            if len(data) > 0:
                break
            
    if len(data) > 0:
        return data
    else:
        return None


def get_pixel_size(detection):
    p1 = detection[3][0]
    p2 = detection[3][1]

    p1 = np.array(p1)
    p2 = np.array(p2)
    
    largo = np.linalg.norm(p2-p1)
    return largo
    

def check(path):
    lista = load_path(path)
    for image in lista:
        print(image)
        detect = qr_detect(image)[0]
        mmperpixel = get_pixel_size(detect) / 100.0
        print(mmperpixel)
   