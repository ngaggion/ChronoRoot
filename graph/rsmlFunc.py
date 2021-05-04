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

import xml.etree.ElementTree as ET
from datetime import datetime
import getpass
import re
import numpy as np
from .imageFunc import skeleton_nodes


def createTree(conf, i, images, grafo, ske, ske2):
    tree = createHeader(conf, i, images)
    root = tree.getroot()
    
    _, enodes = skeleton_nodes(ske)
      
    ## Gets the start node of the graph
    v = grafo[0].get_vertices()
    for k in v:
        if grafo[4][k] == "Ini":
            seed = grafo[1][k]
            seed = np.array(seed, dtype='int')
    
    ## Gets the labels of every edge of the main root
    mainRoot = []
    for k in grafo[3]:
        if k[1] == 10:
            mainRoot.append(k[0])
    
    ## We send the number of first order lateral roots for each analyzed image
    _, numberLR = completeRSML(ske2.copy(), seed, enodes, root, mainRoot)
    
    return tree, numberLR


## Load configuration values for the experiment

def createHeader(conf, i, images):
    rsml_file = "graph/default.rsml"
    tree = ET.parse(rsml_file)
    root = tree.getroot()
    metadata = root[0]
    scene = root[1]   
    
    for elemento in metadata:
        if elemento.tag == 'user':
            elemento.text = getpass.getuser()
    
        if elemento.tag == 'file-key':
            elemento.text = conf['fileKey']
    
        if elemento.tag == 'last-modified':
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%dT%H:%M:%S")
            elemento.text = dt_string
            
        if elemento.tag == 'image':
            for sub in elemento:
                if sub.tag == 'name':
                    sub.text = images[i].replace(conf['Path'],'').replace('/','')
                if sub.tag == 'captured':
                    time = images[i].replace(conf['Path'],'').replace('/','').replace('.png','')
                    year,month,day,hour,mins,secs = re.findall(r'\d+', time)[0:6]
                    text = "%s-%s-%sT%s:%s:%s" %(year, month, day, hour, mins, secs)
                    sub.text = text
    
        if elemento.tag == 'time-sequence':
            for sub in elemento:
                if sub.tag == 'index':
                    sub.text = str(i)
                if sub.tag == 'label':
                    sub.text = conf['sequenceLabel']
    
    ##Adds the analyzed root to the scene
                    
    tag = 'plant'
    attrib = {}
    attrib['id'] = str(1)
    attrib['label'] = conf['Plant']
    
    plant = ET.Element(tag,attrib)
    plant.text = "\n\t\t"
    plant.tail = "\n\t"
    
    scene.append(plant)

    return tree


def completeRSML(ske2, seed, enodes, rsml, mainRoot):
    hijos = vecinos(ske2, seed)
    
    ## Adds the main root to the plant
    
    plant = rsml[1][0]
    tag = 'root'
    attrib = {}
    attrib['id'] = 'p'
    attrib['label'] = 'mainRoot'
    raiz = ET.Element(tag, attrib)
    raiz.text = '\n\t\t\t'
    raiz.tail = '\n\t\t'
    plant.append(raiz)
    geometry = ET.Element('geometry')
    geometry.text = '\n\t\t\t'
    geometry.tail = '\n\t\t'
    raiz.append(geometry)
    polyline = ET.Element('polyline')
    polyline.text = '\n\t\t\t\t'
    polyline.tail = '\n\t\t\t'
    
    tag = 'point'
    attrib = {}
    attrib['x'] = str(seed[0])
    attrib['y'] = str(seed[1])
    node = ET.Element(tag, attrib)
    node.tail = '\n\t\t\t\t'
    polyline.append(node)
    
    ## Gets the next node point
    ske2[seed[1], seed[0]] = 0
    ske2, nodo, largo_arista, polyline = get_next_node_rsml(ske2, hijos[0], seed, [], 0, polyline)
    ske2[nodo[1], nodo[0]] = 0
    geometry.append(polyline)
    
    lista = []
    
    # print(mainRoot)
    
    ## Completes the main Root First, adding every non main root start point to a list
    if largo_arista != 0:
        if nodo not in enodes.tolist():
            rsml, lista = continue_mainRoot(ske2, nodo, enodes, rsml, mainRoot, [])
        
    plant = rsml[1][0][0]
    
    listab = []
    lista_r = []
    num_raices_laterales = 0
    
    largo_lista = len(lista)
    ## 1ST ORDER LATERAL ROOTS
    j = 0
    while j != largo_lista:
        hijos = vecinos(ske2, lista[j])
        
        for i in range(0, len(hijos)):
            if hijos[i] not in lista:

                polyline = ET.Element('polyline')
                polyline.text = '\n\t\t\t\t'
                polyline.tail = '\n\t\t\t'
                
                ske2, nodo, largo_arista, polyline = get_next_node_rsml(ske2, hijos[i], lista[j], hijos, 0, polyline)
                ske2[nodo[1], nodo[0]] = 0

                if largo_arista != 0:
                    num_raices_laterales += 1
                    
                    tag = 'root'
                    attrib = {}
                    attrib['id'] = str(j)
                    attrib['label'] = 'raiz lateral' + str(j)
                    raiz = ET.Element(tag, attrib)
                    raiz.text = '\n\t\t\t'
                    raiz.tail = '\n\t\t'
                    plant.append(raiz)
                    geometry = ET.Element('geometry')
                    geometry.text = '\n\t\t\t'
                    geometry.tail = '\n\t\t'
                    raiz.append(geometry)
                    geometry.append(polyline)
                    
                    if nodo not in enodes.tolist() and nodo not in lista:
                        listab.append(nodo)
                        lista_r.append(raiz)
                else:
                    vec = vecinos(ske2, nodo)
                    for k in range(0, len(vec)):
                        if vec[k] not in lista:
                            lista.append(vec[k])    
                            largo_lista += 1
        j += 1
    
    ## 2ND ORDER (AND HIGHER) LATERAL ROOTS
    c = 0
    while(listab != []):
        c = c + 1
        lista = listab
        plants = lista_r
        listab = []
        lista_r = []
        
        for j in range(0,len(lista)):
            hijos = vecinos(ske2, lista[j])
            
            for i in range(0, len(hijos)):
                if hijos[i] not in lista:
                    polyline = ET.Element('polyline')
                    polyline.text = '\n\t\t\t\t'
                    polyline.tail = '\n\t\t\t'
                    
                    ske2, nodo, largo_arista, polyline = get_next_node_rsml(ske2, hijos[i], lista[j], hijos, 0, polyline)
                    ske2[nodo[1], nodo[0]] = 0
                    
                    if largo_arista != 0:
                        tag = 'root'
                        attrib = {}
                        attrib['id'] = str(j)
                        attrib['label'] = 'raiz lateral' + str(j) + '_' + str(c)
                        raiz = ET.Element(tag, attrib)
                        raiz.text = '\n\t\t\t'
                        raiz.tail = '\n\t\t'
                        geometry = ET.Element('geometry')
                        geometry.text = '\n\t\t\t'
                        geometry.tail = '\n\t\t'
                        raiz.append(geometry)
                        geometry.append(polyline)
                        
                        plants[j].append(raiz)
                    
                        if nodo not in enodes.tolist() and nodo not in lista:
                            listab.append(nodo)
                            lista_r.append(plants[j])
                    else:
                        vec = vecinos(ske2, nodo)
                        for k in range(0, len(vec)):
                            if vec[k] not in lista:
                                listab.append(vec[k])
                                lista_r.append(plants[j])
                        
    return rsml, num_raices_laterales


def continue_mainRoot(ske2, actual, enodes, rsml, mainRoot, lista):
    hijos = vecinos(ske2, actual)
        
    ## Primero me fijo si no me sobró un pixel del color que venia
    for i in range(0, len(hijos)):
        x = hijos[i][0]
        y = hijos[i][1]
        color = ske2[y,x]
        if color == ske2[actual[1], actual[0]]:
            ske2[actual[1], actual[0]] = 0
            actual = hijos[i]
            hijos = vecinos(ske2, actual)
            break
    
    ## Guardo en una lista los que no pertenezcan a la main root para visitarlos luego
    for i in range(0, len(hijos)):
        x = hijos[i][0]
        y = hijos[i][1]
        color = ske2[y,x]
        if color not in mainRoot:
            if actual not in lista:
                lista.append(actual)
    
    ## Continuo la raíz principal
    for i in range(0, len(hijos)):
        x = hijos[i][0]
        y = hijos[i][1]
        color = ske2[y,x]
        
        if color in mainRoot:
            # print(color)
            polyline = ET.Element('polyline')
            polyline.text = '\n\t\t\t\t'
            polyline.tail = '\n\t\t\t'
            
            tag = 'point'
            attrib = {}
            attrib['x'] = str(actual[0])
            attrib['y'] = str(actual[1])
            node = ET.Element(tag, attrib)
            node.tail = '\n\t\t\t\t'
            polyline.append(node)
            
            ske2, nodo, largo_arista, polyline = get_next_node_rsml(ske2, hijos[i], actual, hijos, 0, polyline)
            ske2[nodo[1], nodo[0]] = 0
            
            if largo_arista != 0:
                main_p = rsml[1][0][0][0][0]
                for punto in polyline:
                    main_p.append(punto)
                if nodo not in enodes.tolist():
                    rsml, lista = continue_mainRoot(ske2, nodo, enodes, rsml, mainRoot, lista)
            else:
                vec = hijos.copy()
                vec.pop(i)
                
                lista = lista + vec
                
                ske2, nodo, largo_arista, polyline = get_next_node_rsml(ske2, nodo, hijos[i], hijos, 0, polyline)
                ske2[nodo[1], nodo[0]] = 0
                
                if nodo not in enodes.tolist():
                    rsml, lista = continue_mainRoot(ske2, nodo, enodes, rsml, mainRoot, lista)
                
            
    return rsml, lista


## EXTRA FUNCTIONS
def vecinos(ske, seed): #NEIGHBOURS OF A PIXEL
    lista = []
    x = int(seed[0])
    y = int(seed[1])
    
    ymax = ske.shape[0]
    xmax = ske.shape[1]
    
    for i in range (y+1,y-2,-1):
        for j in range(x-1,x+2):
            if i < ymax and j < xmax:
                if ske[i,j] != 0:
                    if x!=j or y!=i:
                        lista.append([j,i])
    return lista

## WALK A BRANCH TO FIND A NODE
def get_next_node_rsml(ske, actual, padre, hermanos, d, polyline):
    hijos = vecinos(ske, actual)
    sons = []

    tag = 'point'
    attrib = {}
    attrib['x'] = str(actual[0])
    attrib['y'] = str(actual[1])
    node = ET.Element(tag, attrib)
    node.tail = '\n\t\t\t\t'
    polyline.append(node)
    
    for j in hijos:
        if not np.array_equal(j, padre) and not(j in hermanos):
            sons.append(j)
                   
    if len(sons) != 1:
        return [ske, actual, d, polyline]
    
    ske[actual[1], actual[0]] = 0
    hijo = sons[0]
    dist = np.linalg.norm(np.array(actual) - np.array(hijo)) + d
        
    return get_next_node_rsml(ske, hijo, actual, hijos, dist, polyline)
