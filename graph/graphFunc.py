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
import graph_tool.all as gt

c = 3

def createGraph(img, seed, enodes, bnodes):
    g = gt.Graph(directed = False)
    enodes = np.array(enodes)
    bnodes = np.array(bnodes)
    
    pos = g.new_vertex_property("vector<double>")
    nodetype = g.new_vertex_property("string")
    age = g.new_vertex_property("int")
    weight = g.new_edge_property("float")
    clase = g.new_edge_property("vector<int>") #[color, tipo]

    global c
    c = 3
    
    s1, enodes, d = find_nearest(seed, enodes)
        
    semilla = g.add_vertex()
    pos[semilla] = s1
    img[s1[1], s1[0]] = c
    
    hijos = vecinos(img, s1)

    img, nodo, largo_arista = get_next_node(img, hijos[0], s1, [], 0)
    
    n2 = g.add_vertex()    
    pos[n2] = nodo
    arista = g.add_edge(semilla, n2)
    weight[arista] = largo_arista
    clase[arista] = [c, 0]
    
    c = c+1

    if nodo not in enodes.tolist():
        g, pos, weight = continue_graph(g, pos, weight, clase, img, nodo, n2, enodes, bnodes)

    v = g.get_vertices()
    if len(v) < 2:
        raise Exception("Only one vertex")

    for i in v:
        age[i] = 0
        nodetype[i] = "null"
 
    grafo = [g, pos, weight, clase, nodetype, age]
    
    return grafo, s1, img


def continue_graph(g, pos, weight, clase, img, actual, padre, enodes, bnodes):
    # padre refiere al indice en el grafo de graph_tool
    # actual es la posición del nodo en la imagen
    global c
    
    hijos = vecinos(img, actual)
    
    for i in hijos:
        img, nodo, largo_arista = get_next_node(img, i, actual, hijos, 0)
               
        s = gt.find_vertex(g, pos,nodo)
        if s == []:
            s = g.add_vertex()
            pos[s] = nodo
        else:
            s = s[0]
            
        arista = g.add_edge(padre, s)

        weight[arista] = largo_arista
        clase[arista] = [c, 0]
        
        c = c+1
        
        if img[nodo[1],nodo[0]] == 1:
            img[nodo[1], nodo[0]] = c
            
        if nodo not in enodes.tolist():
            g, pos, weight = continue_graph(g, pos, weight, clase, img, nodo, s, enodes, bnodes)           

    return g, pos, weight


def get_next_node(img, actual, padre, hermanos, d):
    global c
    hijos = vecinos(img, actual)
    sons = []
    
    for j in hijos:
        if not np.array_equal(j, padre) and not(j in hermanos):
            sons.append(j)
    
    if d > 0:
        img[actual[1], actual[0]] = c
    
    if len(sons) != 1:
        return [img, actual, d]
    
    img[actual[1], actual[0]] = c
    
    hijo = sons[0]
    dist = np.linalg.norm(np.array(actual) - np.array(hijo)) + d
    
    return get_next_node(img, hijo, actual, hijos, dist)


def vecinos(img, seed, comp = 1):
    lista = []
    x = seed[0]
    y = seed[1]
    
    ymax = img.shape[0]
    xmax = img.shape[1]
    
    for i in range (y+1,y-2,-1):
        for j in range(x-1,x+2):
            if i < ymax and j < xmax:
                if img[i,j] == comp:
                    if x!=j or y!=i:
                        lista.append([j,i])
    return lista


def find_nearest(node, lnodes): #RETURNS THE LIST OF NODES WITHOUT THE NEAREST ONE
    d = np.linalg.norm(node-lnodes, axis = 1)
    p = np.argmin(d)
    nearest = lnodes[p,:]
    lnodes = np.delete(lnodes, p, axis = 0)

    return nearest, lnodes, d


def saveGraph(grafo, path):
    if grafo is not False:
        g, pos, weight, clase, nodetype, age = grafo
        g.vertex_properties["pos"] = pos
        g.vertex_properties["nodetype"] = nodetype
        g.vertex_properties["age"] = age
        g.edge_properties["weight"] = weight
        g.edge_properties["clase"] = clase
        g.save(path)
    else:
        print('Not valid graph')
    return


def saveProps(image, it, grafo, csv_writer, number_lateral_roots):
    if grafo is not False:
        g, pos, weight, clase, nodetype, age = grafo
        
        main_root_len = 0
        sec_root_len = 0
        tot_len = 0
                    
        for i in g.edges():
            tot_len += weight[i]
            if clase[i][1] == 10:
                main_root_len += weight[i]
            else:
                sec_root_len += weight[i]
        
        row = [image, it, main_root_len, sec_root_len, number_lateral_roots, tot_len]
    else:
        row = [image, it, 0, 0, 0, 0]
        
    csv_writer.writerow(row)
    return
