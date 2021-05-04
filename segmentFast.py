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

def SaveSegImage(conf, name, segmentation, path, suffix = ".png"):    
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

def SegmentUNet(conf, input_dir, output_dir, crf):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    Provider = DataProvider(input_dir, data_suffix = ".png")
    data, name = Provider(1)

    sess = tf.compat.v1.Session()

    conf["batchSize"] = 1
    conf["tileSize"] = list(data.shape[1:3])

    net = RootNet(sess, conf, "RootNET", True)
    
    conf['ckptDir'] = os.path.join(os.path.join('modelWeights', conf['SavePoint']),'ckpt')
    net.restore(conf['ckptDir'])

    n = len(Provider.data_files)
    accum = np.zeros(data.shape[1:3])
    
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
                    
            accum = conf['Alpha'] * accum + crf_map[:,:,1]
        else:
            accum = conf['Alpha'] * accum + segment[0,:,:,1]
        
        _, outimg = cv2.threshold(accum, conf['Thresh'], 1.0, cv2.THRESH_BINARY)
        SaveSegImage(conf, name, outimg, output_dir, ".png")
        
    tf.compat.v1.reset_default_graph()
    sess.close()    
    print("Session ended succesfully")


if __name__ == "__main__":
    conf = {}
    file = exec(open('config.conf').read(), conf)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="ResUNetDS", metavar='M', help="The trained model to use (default ResUNetDS)")
    parser.add_argument('--use_crf', action='store_true', default=False, help='no CRF post-processing')
    parser.add_argument('--output_dir', type=str, help='Output directory', nargs="?")
    parser.add_argument('input_dir', type=str, help='Input directory', nargs="?")

    args = parser.parse_args()
    
    available_models = ['UNet', 'ResUNet', 'ResUNetDS', 'SegNet', 'DeepLab']
    if args.model not in available_models:
        conf['Model'] = 'ResUNetDS'
        conf['SavePoint'] = args.model
        #raise Exception()
    else:
        conf['SavePoint'] = args.model
        conf['Model'] = args.model    

    if not args.input_dir:
        parser.print_help()
        raise Exception()
    
    if not args.output_dir:
        output_dir = os.path.join(args.input_dir, 'Seg')
    else:
        output_dir = args.output_dir
          
    try:
        mkdir(output_dir)
    except:
        pass

    use_crf = args.use_crf

    try:
        SegmentUNet(conf, args.input_dir, output_dir, use_crf)
    except:
        raise Exception()
    


