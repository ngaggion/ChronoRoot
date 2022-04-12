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

import os
import pathlib
import re
import cv2
import numpy as np

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def loadPath(search_path, ext = '*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key = natural_key)
    
    return all_files


def createResultFolder(conf):
    try:
        os.mkdir(conf['Project'])
    except:
        pass

    for i in range(0,100):
        saveFolder = os.path.join(conf['Project'], "Results %s" %(i))
        try:
            os.mkdir(saveFolder)
            break
        except:
            if i == 20:
                raise Exception('Could not create Results Folder')
            pass
    
    graphsPath =  os.path.join(saveFolder, "Graphs")
    try:
        os.mkdir(graphsPath)
    except:
        pass
    
    if conf['SaveImages']:
        imagePath = os.path.join(saveFolder, "Imagenes")
        try:
            os.mkdir(imagePath)
            
            f1 = os.path.join(imagePath, "img")
            f2 = os.path.join(imagePath, "seg")
            f3 = os.path.join(imagePath, "labeledSeg")
            f4 = os.path.join(imagePath, "graph")
            
            os.mkdir(f1)
            os.mkdir(f2)
            os.mkdir(f3)
            os.mkdir(f4)
        except:
            pass
        
    rsmlPath = os.path.join(saveFolder, "RSML")
    try:
        os.mkdir(rsmlPath)
    except:
        pass
    
    return saveFolder, graphsPath, imagePath, rsmlPath


def selectROI(image):
    y, x = image.shape[0:2]
    y = y//4
    x = x//4
    
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", x, y)
    
    r = cv2.selectROI("Image",image)

    cv2.waitKey()
    cv2.destroyAllWindows()
    return r

pos = []


def mouse_callback(event,x,y,flags,param):
    global pos
    if event == cv2.EVENT_LBUTTONDOWN:
        pos = [(x, y)]
        print(pos)
        
def selectSeed(images):
    n = len(images)
    i = 0
    image = images[i]
    
    global pos
    y, x = image.shape[0:2]
    y = y//2
    x = x//2
    
    clone = image.copy()
    
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", x, y)
    cv2.setMouseCallback('Image',mouse_callback) #Mouse callback
    
    while True:
        if pos != []:
            cv2.circle(clone,pos[0],6,[255,0,0],-1)
        
    	# display the image and wait for a keypress
        cv2.imshow("Image", clone)
        key = cv2.waitKey(1) & 0xFF
     
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("+"):
            i = (i+1)%n
            image = images[i]
            clone = image.copy()
        
        if key == ord("r"):
            clone = image.copy()
     
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
        
        elif key == 13:
            break
    
    cv2.destroyAllWindows()
    return pos


def getROIandSeed(conf, images, segFiles):
    N1 = len(images)
    N2 = len(segFiles)

    N = min(N1,N2)
    
    original = cv2.imread(images[N-1])
    
    r = selectROI(original)
    p = np.array([int(r[1]),int(r[1]+r[3]), int(r[0]),int(r[0]+r[2])])
    
    crops = []
    
    t = conf['timeStep'] #always in minutes
    dia = 24*60/t
    c = int(N//dia)
    c = max(1, c)
    
    for i in range(0, c):
        P2 = int(i*100)
        img = cv2.imread(images[P2])
        boundingBox = img[p[0]:p[1],p[2]:p[3]]
        crops.append(boundingBox)
    
    seed = selectSeed(crops)
    
    return p, seed