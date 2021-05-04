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

import numpy as np
import re
import pandas as pd
import os
from .qr import get_pixel_size, qr_detect, load_path
from scipy import signal

def dataWork(conf, pfile, folder):
    data = pd.read_csv(pfile)
    shape = data.shape
    N = shape[0]
    
    path = os.path.abspath(os.path.join(folder, '..'))
    files = load_path(conf['Path'], '*.png')
    files = [file for file in files if 'mask' not in file][:20]

    for i in files:
        try:
            detect = qr_detect(i)[0]
            pixel_size = 10 / get_pixel_size(detect)
            break
        except:
            pixel_size = 0.04
            pass
    
    print('Pixel size (in mm): ', pixel_size)
    
    index = data['TimeStep'].to_numpy()
    mainRoot = data['MainRootLength'].to_numpy().astype('float')
    lateralRoots = data['LateralRootsLength'].to_numpy().astype('float')
    numlateralRoots = data['NumberOfLateralRoots'].to_numpy().astype('float')

    space = 30
    for t in range(space, N//2):
        if numlateralRoots[t-space] == 0 and numlateralRoots[t] == 0:
            lateralRoots[t-space:t] = 0
            numlateralRoots[t-space:t] = 0

    # lateralRoots[0:600] = 0.0
    # numlateralRoots[0:600] = 0.0

    # Smooth
    mainRoot = signal.medfilt(mainRoot, 5) 
    lateralRoots = signal.medfilt(lateralRoots, 5) 
    numlateralRoots = signal.medfilt(numlateralRoots, 25)

    for i in range(1, len(mainRoot)):
        # dif = mainRoot[i] < mainRoot[i-1]
        # if dif:
        #     d = mainRoot[i-1] - mainRoot[i]
        #     if d < 15:
        #         mainRoot[i] = mainRoot[i-1]
        
        dif = numlateralRoots[i] < numlateralRoots[i-1]
        if dif and numlateralRoots[i-1] > 4:
            numlateralRoots[i] = numlateralRoots[i-1]

        dif = lateralRoots[i] < lateralRoots[i-1]
        if dif and lateralRoots[i-1] > 50:
            lateralRoots[i] = lateralRoots[i-1]

    mainRootProcessed = mainRoot.copy().astype('float') 
    lateralRootsProcessed = lateralRoots.copy().astype('float') 
    numlateralRootsProcessed = numlateralRoots.copy().astype('float')

    r = 4
    for j in range(r, len(mainRootProcessed)):
        mainRootProcessed[j-r:j+r+1] = np.mean(mainRoot[j-r:j+r+1])
        lateralRootsProcessed[j-r:j+r+1] = np.mean(lateralRoots[j-r:j+r+1])

    mainRootProcessed = mainRootProcessed * pixel_size
    lateralRootsProcessed = lateralRootsProcessed * pixel_size

    time = np.zeros(index.shape, dtype='float')
    for i in range(0,N):
        name = data['FileName'][i]
        nums = re.findall(r'\d+', name)
        hora = int(nums[3])
        minutos = int(nums[4])
        time[i] = hora + minutos / 100
        
    timehours = index * int(conf['timeStep']) / 60
    
    data.insert(data.shape[1], 'Time elapsed (hours)', timehours)
    data['MainRootLength'] = mainRootProcessed
    data['LateralRootsLength'] = lateralRootsProcessed
    data['NumberOfLateralRoots'] = numlateralRootsProcessed
    data['TotalLength'] = mainRootProcessed + lateralRootsProcessed
    
    data.insert(data.shape[1], 'Acquisition Time', time)

    n = int(np.floor(N/2))
    n2 = int(np.floor(n/2))

    v = np.zeros(index.shape, dtype='bool')

    for i in range(1, shape[0]):
        if time[i] < time[i-1]:
            v[i] = True
        else:
            v[i] = False
            
    v1 = np.zeros(n, dtype = 'bool') 
    for i in range(0,n):
        v1[i] = np.any(v[2*i:2*i+2])
        
    v2 = np.zeros(n2, dtype = 'bool')
    for i in range(0,n2):
        v2[i] = np.any(v1[2*i:2*i+2])

    mainRootPooled1 = np.zeros(n, dtype='float')    
    lateralRootsPooled1 = np.zeros(n, dtype='float')
    numlateralRootsPooled1 = np.zeros(n, dtype='float')

    for i in range(0,n):
        mainRootPooled1[i] = np.mean(mainRootProcessed[2*i:2*i+2])
        lateralRootsPooled1[i] = np.mean(lateralRootsProcessed[2*i:2*i+2])
        numlateralRootsPooled1[i] = np.mean(numlateralRootsProcessed[2*i:2*i+2])        

    mainRootPooled2 = np.zeros(n2, dtype='float')    
    lateralRootsPooled2 = np.zeros(n2, dtype='float')
    numlateralRootsPooled2 = np.zeros(n2, dtype='float')

    for i in range(0,n2):
        mainRootPooled2[i] = np.mean(mainRootPooled1[2*i:2*i+2])
        lateralRootsPooled2[i] = np.mean(lateralRootsPooled1[2*i:2*i+2])
        numlateralRootsPooled2[i] = np.mean(numlateralRootsPooled1[2*i:2*i+2])


    mainRootGrad = np.gradient(mainRootPooled2, edge_order = 2)
    lateralRootsGrad = np.gradient(lateralRootsPooled2, edge_order = 2)
    
    mainRootGrad = signal.medfilt(mainRootGrad, 5)
    lateralRootsGrad = signal.medfilt(lateralRootsGrad, 5)
    
    totalRootsLengthPooled = mainRootPooled2 + lateralRootsPooled2
    totalRootsGrad = np.gradient(totalRootsLengthPooled, edge_order = 2)
    totalRootsGrad = signal.medfilt(totalRootsGrad, 5)
    
    aporte_al_total = mainRootPooled2 / totalRootsLengthPooled * 100
    where_are_NaNs = np.isnan(aporte_al_total)
    aporte_al_total[where_are_NaNs] = 100.0
    np.clip(aporte_al_total, 0, 100)
    
    aporte_al_total = signal.medfilt(aporte_al_total, 5)
    
    densidad_lateral = 10 * numlateralRootsPooled2 / mainRootPooled2
    where_are_NaNs = np.isnan(densidad_lateral)
    densidad_lateral[where_are_NaNs] = 0.0
    
    densidad_lateral = signal.medfilt(densidad_lateral, 5)
    
    densidad_lateral_continua = lateralRootsPooled2 / mainRootPooled2
    where_are_NaNs = np.isnan(densidad_lateral_continua)
    densidad_lateral_continua[where_are_NaNs] = 0.0
    
    densidad_lateral_continua = signal.medfilt(densidad_lateral_continua, 5)
    
    pooledData = pd.DataFrame(data={'Time': np.arange(mainRootGrad.shape[0]),
                                    'mainRootLength': mainRootPooled2,
                                    'lateralRootsLength': lateralRootsPooled2,
                                    'totalRootsLength': totalRootsLengthPooled,
                                    'mainRootGrad': mainRootGrad,
                                    'lateralRootsGrad': lateralRootsGrad,
                                    'totalRootsGrad': totalRootsGrad,
                                    'NumberOfLateralRoots': numlateralRootsPooled2,
                                    'newDay' : v2,
                                    'mainOverTotal' : aporte_al_total,
                                    'lateralRootDensity' : densidad_lateral,
                                    'lateralRootContDensity' : densidad_lateral_continua
                                    })
    
    pooledData = np.clip(pooledData, 0, 1e10)
    
    data.to_csv(os.path.join(folder, 'Postprocessed.csv'), index = False)
    pooledData.to_csv(os.path.join(folder, 'GrowthSpeeds.csv'), index = False)
    
    return       
 
