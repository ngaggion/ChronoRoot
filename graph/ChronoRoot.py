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
import csv
import cv2
import numpy as np
import json

from .fileFunc import createResultFolder, loadPath, getROIandSeed
from .imageFunc import getCleanSeg, getCleanSke, savePlotImages, saveEmpty
from .graphFunc import createGraph, saveGraph, saveProps
from .trackFunc import graphInit, matchGraphs
from .rsmlFunc import createTree
from .graphPostProcess import trimGraph
from .dataWork import dataWork

def getImgName(image, conf):
    return image.replace(conf['Path'],'').replace('/','')

def ChronoRootAnalyzer(conf):
    ext = "*" + conf["FileExt"]
    all_files = loadPath(conf['Path'], ext) 
    images = [file for file in all_files if 'mask' not in file]
       
    ext = "*" + conf["FileExt"]
    all_files = loadPath(conf['SegPath'], ext) 
    segFiles = [file for file in all_files if 'mask' in file]
    
    lim = conf['Limit'] 
    
    if lim!=0:
        images = images[:lim]
        segFiles = segFiles[:lim]
    
    bbox, seed = getROIandSeed(conf, images, segFiles)
    seed = list(seed[0])
    originalSeed = seed.copy()
    
    saveFolder, graphsPath, imagePath, rsmlPath = createResultFolder(conf)
    
    metadata = {}
    metadata['bounding box'] = bbox.tolist()
    metadata['seed'] = seed
    metadata['folder'] = conf['Path']
    metadata['segFolder'] = conf['SegPath']
    metadata['info'] = conf['fileKey']

    print(metadata)
    metapath = os.path.join(saveFolder, 'metadata.json')

    with open(metapath, 'w') as fp:
        json.dump(metadata, fp)

    start = 0
    N = len(images)
    pfile = os.path.join(saveFolder, "Results.csv") # For CSV Saver
    
    with open(pfile, 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        row0 = ['FileName', 'TimeStep','MainRootLength','LateralRootsLength','NumberOfLateralRoots','TotalLength']
        csv_writer.writerow(row0)
        
        ### First, it begins by obtaining the first segmentation

        for i in range(0, N):
            print('TimeStep', i+1, 'of', N)
            segFile = segFiles[i]
            seg, segFound = getCleanSeg(segFile, bbox, originalSeed, originalSeed)
            
            original = cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]
            
            if segFound:
                ske, bnodes, enodes, flag = getCleanSke(seg)
                if flag:
                    start = i
                    break
            
            image_name = getImgName(images[i], conf)
            saveProps(image_name, i, False, csv_writer, 0)
            saveEmpty(image_name, imagePath, original, seg)
        
        print('Growth Begin')
        
        grafo, seed, ske2 = createGraph(ske.copy(), seed, enodes, bnodes)
        grafo, ske, ske2 = trimGraph(grafo, ske, ske2)
        grafo = graphInit(grafo)
        
        image_name = getImgName(images[i], conf)
        
        gPath = os.path.join(graphsPath, image_name.replace(conf['FileExt'],'.xml.gz'))
        saveGraph(grafo, gPath)
        
        rsmlTree, numberLR = createTree(conf, i, images, grafo, ske, ske2)
        
        rsml = os.path.join(rsmlPath, image_name.replace(conf['FileExt'],'.rsml'))
        rsmlTree.write(open(rsml, 'w'), encoding='unicode')        
        
        saveProps(image_name, i, grafo, csv_writer, numberLR)
        
        original = cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        savePlotImages(image_name, imagePath, original, seg, grafo, ske2)
        
        segErrorFlag = False #Previous time-step error
        trackCount = 0
        
        for i in range(start+1, N):
            print('TimeStep', i+1, 'of', N)
            errorFlag_ = False
            
            segFile = segFiles[i]
            seg, flag1 = getCleanSeg(segFile, bbox, seed.tolist(), originalSeed)
            
            if flag1:
                ske, bnodes, enodes, flag2 = getCleanSke(seg)
                if not flag2:
                    print("Error in the skeleton")
                    errorFlag_ = True
            else:
                print("Error in the segmentation")
                errorFlag_ = True
            
            trackError = False
        
            if not errorFlag_:               
                grafo2, seed, ske2_ = createGraph(ske.copy(), seed, enodes, bnodes)
                grafo2, ske_, ske2_ = trimGraph(grafo2, ske.copy(), ske2_)
                
                if not segErrorFlag:
                    try:
                        grafo = matchGraphs(grafo, grafo2)
                        ske =  ske_.copy()
                        ske2 = ske2_.copy()
                    except:
                        print("Error on node tracking")
                        trackError = True
                else:
                    grafo = graphInit(grafo2)
                    ske =  ske_.copy()
                    ske2 = ske2_.copy()
                    
            else:
                image_name = getImgName(images[i], conf)
                saveProps(image_name, i, False, csv_writer, 0)
                saveEmpty(image_name, imagePath, original, seg)
            
            segErrorFlag = errorFlag_
                        
            if not segErrorFlag and not trackError:           
                gPath = os.path.join(graphsPath, image_name.replace(conf['FileExt'],'.xml.gz'))
                saveGraph(grafo, gPath)
        
                seedrsml = None
                v = grafo[0].get_vertices()
                for k in v:
                    if grafo[4][k] == "Ini":
                        seedrsml = grafo[1][k]
                        seedrsml = np.array(seed, dtype='int')
                
                if seedrsml is None:
                    trackError = True
                    image_name = images[i].replace(conf['Path'],'').replace('/','')
                    saveProps(image_name, i, False, csv_writer, 0)
                    saveEmpty(image_name, imagePath, original, seg)
                else:
                    rsmlTree, numberLR = createTree(conf, i, images, grafo, ske, ske2)
                    rsml = os.path.join(rsmlPath, image_name.replace(conf['FileExt'],'.rsml'))
                    rsmlTree.write(open(rsml, 'w'), encoding='unicode')        
        
                    image_name = getImgName(images[i], conf)
                    saveProps(image_name, i, grafo, csv_writer, numberLR)
                    
                    original = cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]
                    savePlotImages(image_name, imagePath, original, seg, grafo, ske2)
        
            if trackError and trackCount > 5:
                print('Analysis ended early at timestep', i, 'of', N)
                break
            elif trackError:
                trackCount += 1
            else:
                trackCount = 0
    
    dataWork(conf, pfile, saveFolder)