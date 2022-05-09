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

import tensorflow as tf
import os
import numpy as np

from rootNet.Model import RootNet
from rootNet.Provider import DataProvider

import nibabel as nib
import cv2
import argparse
import pydensecrf.densecrf as dcrf

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import re
import pathlib


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def loadPath(search_path, ext = '*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key = natural_key)
    
    return all_files


def mkdir(dir_path):
    try :
      os.makedirs(dir_path)
    except: pass 


def save_image_as_it_is(path, arr):
    cv2.imwrite(path,arr)  

      
def save_image_with_scale(path, arr):
    arr = np.clip(arr, 0., 1.)
    arr = arr * 255.
    arr = arr.astype(np.uint8)
    cv2.imwrite(path,arr)


def padImgToMakeItMultipleOf(v, multipleOf=[8, 8], mode='symmetric'):
    padding = ((0, 0 if v.shape[0] % multipleOf[0] == 0 else multipleOf[0] - (v.shape[0] % multipleOf[0])),
               (0, 0 if v.shape[1] % multipleOf[1] == 0 else multipleOf[1] - (v.shape[1] % multipleOf[1])))
    return np.pad(v, padding, mode)


def SaveSegImage(conf, name, segmentation, path, suffix = ".png", cutpad = False):    
    #cutpad removes padding after segmentation
    if cutpad:
        h, w = conf['OriginalSize']
        segmentation = segmentation[:h, :w]
            
    if suffix == ".nii.gz":
        name = name[0][0].replace(suffix,".nii.gz")
        nombre = os.path.join(path, name)
        img = nib.Nifti1Image(segmentation.transpose(), np.eye(4))
        nib.save(img, nombre)
    else:   
        name = name[0][0].replace(suffix,"_mask.png")
        nombre = os.path.join(path, name)
        save_image_with_scale(nombre, segmentation)
    return


def Segment(conf, input_dir, output_dir, crf):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    Provider = DataProvider(input_dir, data_suffix = ".png")
    data, name = Provider(1)
    
    sess = tf.compat.v1.Session()

    conf["batchSize"] = 1
    conf["tileSize"] = list(data.shape[1:3])

    net = RootNet(sess, conf, "RootNET", False)
    
    conf['ckptDir'] = os.path.join(os.path.join('modelWeights', conf['Model']),'ckpt')
    net.restore(conf['ckptDir'])

    limit = conf['LIMIT']

    if limit != -1:
        n = limit
    else:
        n = len(Provider.data_files)
    
    for i in range(0, n):
        print("File %s out of %s" %(i+1,n))

        if i!=0:
            data, name = Provider(1)
            
        segment = net.segment(data)

        if crf:
            image = cv2.cvtColor((data[0,:,:,0]*255).astype('uint8'), cv2.COLOR_GRAY2RGB)
            image = np.ascontiguousarray(image)
            
            label_1 = np.transpose(segment[0,:,:,:], (2,0,1))
            unary = -np.log(np.clip(label_1,1e-5,1.0))
            c, h, w = unary.shape
            unary = unary.transpose(0, 2, 1)
            unary = unary.reshape(2, -1)
            unary = np.ascontiguousarray(unary)
            
            denseCRF = dcrf.DenseCRF2D(w, h, 2)
            denseCRF.setUnaryEnergy(unary)  
            denseCRF.addPairwiseBilateral(sxy=5, srgb=3, rgbim=image, compat=1)
            
            q = denseCRF.inference(1)
            crf_map = np.array(q).reshape(2, w, h).transpose(2, 1, 0)
        
            outimg = crf_map[:,:,1]
        else:
            outimg = segment[0,:,:,1]
                    
        SaveSegImage(conf, name, outimg, output_dir, ".png", True)
        
    tf.compat.v1.reset_default_graph()
    sess.close()    
    print("Session ended succesfully")


def ensembleModels(conf, input_dir, output_dir, crf, models):   
    outs = []
    for i in models:
        outs.append(os.path.join(output_dir, i))
    
    images = loadPath(input_dir, '*.png') 

    accum = np.zeros(cv2.imread(images[0], 0).shape, dtype=float)
    
    limit = conf['LIMIT']

    if limit != -1:
        n = limit
    else:
        n = len(images)

    for i in range(0, n):
        print("File %s out of %s" %(i+1,n))
        
        segs = []
        for t in outs:
            path = images[i].replace(input_dir, t).replace('.png','_mask.png')
            segs.append(cv2.imread(path, 0).astype('float') / 255.0)
        
        ensemble = np.zeros_like(segs[0])
        
        for t in range(0, len(models)):
            ensemble += segs[t]
        
        ensemble = ensemble / len(models)
        
        if crf:
            image = cv2.cvtColor((ensemble*255).astype('uint8'), cv2.COLOR_GRAY2RGB)
            image = np.ascontiguousarray(image)
            
            label_1 = np.transpose(ensemble, (2,0,1))
            unary = -np.log(np.clip(label_1,1e-5,1.0))
            c, h, w = unary.shape
            unary = unary.transpose(0, 2, 1)
            unary = unary.reshape(2, -1)
            unary = np.ascontiguousarray(unary)
            
            denseCRF = dcrf.DenseCRF2D(w, h, 2)
            denseCRF.setUnaryEnergy(unary)  
            denseCRF.addPairwiseBilateral(sxy=5, srgb=3, rgbim=image, compat=1)

            q = denseCRF.inference(1)
            crf_map = np.array(q).reshape(2, w, h).transpose(2, 1, 0)
        
            accum = conf['Alpha'] * accum + crf_map[:,:,1]
        else:
            accum = conf['Alpha'] * accum + ensemble
            
        _, outimg = cv2.threshold(accum, conf['Thresh'], 1.0, cv2.THRESH_BINARY)
        SaveSegImage(conf, [[images[i].replace(input_dir, '').replace('/','')]], outimg, output_dir, ".png", True)
     
    return
    

if __name__ == "__main__":
    conf1 = {}
    file = exec(open('config.conf').read(), conf1)
    conf2 = {}
    file = exec(open('cnns.conf').read(), conf2)
    
    conf = {**conf1, **conf2}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_crf', action='store_true', default=False, help='no CRF post-processing')
    parser.add_argument('--output_dir', type=str, help='Output directory', nargs="?")
    parser.add_argument('input_dir', type=str, help='Input directory', nargs="?")

    args = parser.parse_args()
    
    if not args.input_dir:
        parser.print_help()
        raise Exception()
    
    if not args.output_dir:
        output_dir = os.path.join(args.input_dir, 'SegEnsemble')
    else:
        output_dir = args.output_dir
          
    try:
        mkdir(output_dir)
    except:
        pass

    use_crf = args.use_crf

    available_models = ['UNet', 'ResUNet', 'ResUNetDS', 'SegNet', 'DeepLab']

    for i in available_models:
        conf['Model'] = i  
        out = os.path.join(output_dir, i)
        mkdir(out)
        Segment(conf, args.input_dir, out, False)

    ensembleModels(conf, args.input_dir, output_dir, use_crf, available_models)

    