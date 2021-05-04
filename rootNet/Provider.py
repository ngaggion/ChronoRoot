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

from __future__ import print_function, division, absolute_import, unicode_literals

import pathlib
import numpy as np
from PIL import Image
import cv2
import nibabel as nib
import re

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def padImgToMakeItMultipleOf(v, multipleOf=[8, 8], mode='symmetric'):
    padding = ((0, 0 if v.shape[0] % multipleOf[0] == 0 else multipleOf[0] - (v.shape[0] % multipleOf[0])),
               (0, 0 if v.shape[1] % multipleOf[1] == 0 else multipleOf[1] - (v.shape[1] % multipleOf[1])))
    return np.pad(v, padding, mode)


class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """

    channels = 1
    n_class = 2
    

    def __init__(self, a_min=0, a_max=255):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        train_data, label = self._next_data()       
        labels = self._process_labels(label)  
        return train_data, labels
        
    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels        
        return label
    

    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        X = []
        Y = []
        X.append(train_data)
        Y.append(labels)
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X.append(train_data)
            Y.append(labels)

        # print('All images loaded.')

        return X, Y


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return np.float32(cv2.LUT(image.astype('uint8'), table))

   
class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, search_path, augment = False, random_state = None, a_min=None, a_max=None, 
                 data_suffix=".png", mask_suffix='.nii.gz', shuffle_data=False, n_class = 2):
        super(ImageDataProvider, self).__init__(a_min, a_max)
      
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.augment = augment
        
        self.data_files = self._find_data_files(search_path)
        
        if not(random_state):
            self.random_state = np.random.RandomState(None)
        else:
            self.random_state = random_state
            # print(random_state)
        
        if self.shuffle_data:
            self.random_state.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        # print("Number of files used: %s" % len(self.data_files))
        
        self.channels = 1        
        
    
    def _augment(self, image, label):
        coin = self.random_state.uniform(0,1)
        if coin > 0.6:    
            value = self.random_state.uniform(0.7,1.3)
            image = adjust_gamma(image, value)
        
        coin = self.random_state.uniform(0,1)
        if coin > 0.3:        
            image = cv2.flip(image,1)
            label = cv2.flip(label.astype('uint8'),1).astype('bool')

        coin = self.random_state.uniform(0,1)
        if coin > 0.1:
            image = cv2.GaussianBlur(image,(5,5),0)

        coin = self.random_state.uniform(0,1)
        if coin > 0.2:
            noise = self.random_state.randn(image.shape[0],image.shape[1]).astype('float32')*4
            image = cv2.add(image, noise)
            image = np.clip(image, 0, 255)
            
        # coin = self.random_state.uniform(0,1)
        # if coin > 0.5:
        #     value = self.random_state.uniform(0.90,1.10)
        #     image = cv2.resize(image, (0,0), fx=value, fy=value)
        #     label = cv2.resize(label.astype('float'), (0,0), fx=value, fy=value).astype('bool')
            
        return image, label
    
    
    def _find_data_files(self, search_path):
        data_root = pathlib.Path(search_path)
        all_files = list(data_root.glob('*.*'))
        all_files = [str(path) for path in all_files]
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]
    
    
    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path).convert('L'), dtype)


    def _load_mask(self, path):
        img = nib.load(path)
        img = np.transpose(img.get_fdata()[:,:,0]).astype(np.bool)
        return img
    
    
    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            
        
    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        img = self._load_file(image_name, np.float32)
        label = self._load_mask(label_name)
	
        if self.augment:
            img, label = self._augment(img, label)
    
        return img,label


class MPImageDataProvider(BaseDataProvider):
    
    def __init__(self, search_path, augment = False, random_state = None, a_min=None, a_max=None, 
                 data_suffix=".png", mask_suffix='.nii.gz', shuffle_data=False, n_class = 2):
        super(MPImageDataProvider, self).__init__(a_min, a_max)
      
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.data_files = self._find_data_files(search_path)
        self.augment = augment
        
        if not(random_state):
            self.random_state = np.random.RandomState(None)
        else:
            self.random_state = random_state
        
        if self.shuffle_data:
            self.random_state.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        # print("Number of files used: %s" % len(self.data_files))
        
        self.channels = 1        
        
    
    def _augment(self, image, label):
        coin = self.random_state.uniform(0,1)
        if coin > 0.7:    
            value = self.random_state.uniform(0.7,1.3)
            image = adjust_gamma(image, value)
        
        coin = self.random_state.uniform(0,1)
        if coin > 0.3:        
            image = cv2.flip(image,1)
            label = cv2.flip(label.astype('uint8'),1).astype('bool')

        coin = self.random_state.uniform(0,1)
        if coin > 0.2:
            image = cv2.GaussianBlur(image,(5,5),0)

        coin = self.random_state.uniform(0,1)
        if coin > 0.3:
            noise = self.random_state.randn(image.shape[0],image.shape[1]).astype('float32')*4
            image = cv2.add(image, noise)
            image = np.clip(image, 0, 255)
            
        coin = self.random_state.uniform(0,1)
        if coin > 0.5:
            value = self.random_state.uniform(0.90,1.10)
            image = cv2.resize(image, (0,0), fx=value, fy=value)
            label = cv2.resize(label.astype('float'), (0,0), fx=value, fy=value).astype('bool')
                
        return image, label
    
    def _find_data_files(self, search_path):
        data = []
        for i in search_path:
            data_root = pathlib.Path(i)
            all_files = list(data_root.glob('*.*'))
            all_files = [str(path) for path in all_files]
            for name in all_files: 
                if self.data_suffix in name and not self.mask_suffix in name:
                    data.append(name)
        return data
    
    
    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path).convert('L'), dtype)


    def _load_mask(self, path):
        img = nib.load(path)
        img = np.transpose(img.get_fdata()[:,:,0]).astype(np.bool)
        return img
    
    
    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 

        
    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        img = self._load_file(image_name, np.float32)
        label = self._load_mask(label_name)
	
        if self.augment:
            img, label = self._augment(img, label)
    
        return img,label


class DataProvider(BaseDataProvider):
    
    def __init__(self, search_path, pad = True, random_state = None, a_min=None, a_max=None, data_suffix=".png", mask_suffix='_mask.png', shuffle_data=False, n_class = 2):
        super(DataProvider, self).__init__(a_min, a_max)
      
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.pad = pad
        self.search_path = search_path
        
        self.data_files = self._find_data_files(search_path)
        
        if not(random_state):
            self.random_state = np.random.RandomState(None)
        else:
            self.random_state = random_state
        
        if self.shuffle_data:
            self.random_state.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        # print("Number of files used: %s" % len(self.data_files))
        
        self.channels = 1        
        
        
    def _find_data_files(self, search_path):
        data_root = pathlib.Path(search_path)
        all_files = list(data_root.glob('*.*'))
        all_files = [str(path) for path in all_files]
        all_files.sort(key = natural_key)
        data = [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]
        return [name for name in data if not name.replace(self.data_suffix, self.mask_suffix) in all_files]
    
    
    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path).convert('L'), dtype)
        
    
    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 

        
    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        img = self._load_file(image_name, np.float32)
        
        return img
    
    def __call__(self, n):
        names = []
        train_data = self._next_data() / 255.0
        
        name = self.data_files[self.file_idx]
        name = name.replace(self.search_path,'')
        if name[0] == "/":
            name = name[1:]
        names.append([name])
        
        if self.pad:
            train_data = padImgToMakeItMultipleOf(train_data)
            
        
        ny = train_data.shape[0]
        nx = train_data.shape[1]
        
        X = np.zeros((n, ny, nx, 1))
    
        X[0,:,:,0] = train_data
        for i in range(1, n):
            train_data = self._next_data()
            
            name = self.data_files[self.file_idx]
            name = name.replace(self.search_path,'')
            if name[0] == "/":
                name = name[1:]
            
            names.append([name])
            
            train_data = padImgToMakeItMultipleOf(train_data)
            X[i,:,:,0] = train_data
    
        return X, names
