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

def graphInit(graph):
    g, pos, weight, clase, nodetype, age = graph
    
    vertices = g.get_vertices()
    
    pos_vertex = []
    for i in vertices:
        pos_vertex.append(pos[i])
    pos_vertex = np.array(pos_vertex)
    
    s = np.argmin(pos_vertex[:,1])
    seed = pos_vertex[s,:]
    t = np.argmax(pos_vertex[:,1])
    tip = pos_vertex[t,:]
    
    for i in vertices:
        p1 = pos_vertex[i]
        if np.array_equal(p1, seed):
            nodetype[i] = "Ini"
        else:
            if np.array_equal(p1, tip):
                nodetype[i] = "FTip"
            else:
                vecinos = g.get_out_neighbours(i)
                if len(vecinos) > 1:
                    nodetype[i] = "Bif"
                else:
                    if len(vecinos) == 1:
                        nodetype[i] = "LTip"
                
    if len(vertices) == 2:
        clase[g.edge(s,t)][1] = 10
        age[g.vertex(s)] = 1
        age[g.vertex(t)] = 1

    return [g, pos, weight, clase, nodetype, age]


def find_nearest_b(node, lnodes):
    d = np.linalg.norm(node-lnodes, axis = 1)
    p = np.argmin(d)

    return p, d[p]


def find_nearest_nodes(node, lnodes, thresh = 10): 
    d = np.linalg.norm(node-lnodes, axis = 1)
    nodes = d<thresh
    
    p = nodes.nonzero()[0]

    return p, d[p]


def matchGraphs(graph1, graph2):
    g1, pos1, weight1, clase1, nodetype1, age1 = graph1
    g2, pos2, weight2, clase2, nodetype2, age2 = graph2
    
    vertices1 = g1.get_vertices()
    vertices2 = g2.get_vertices()
    
    age = np.zeros_like(vertices1)
    
    for i in range(0, len(vertices1)):
        age[i] = age1[vertices1[i]]
    
    vertices1 = np.argsort(age)[::-1]
    
    pos_vertex2 = []
    for i in vertices2:
        pos_vertex2.append(pos2[i])
    pos_vertex2 = np.array(pos_vertex2)

    for i in vertices1:
        p1 = pos1[i]
        v, d = find_nearest_nodes(p1, pos_vertex2, 15)
        
        v = v[np.argsort(d)]
        if len(v) == 1:
            age2[v[0]] = age1[i] + 1            
            if nodetype2[v[0]] == "null":
                nodetype2[v[0]] = nodetype1[i]
        elif len(v) == 2:
            n_0 = len(g1.get_out_neighbours(i))
            n_1 = len(g2.get_out_neighbours(v[0]))
            n_2 = len(g2.get_out_neighbours(v[1]))
            
            dif_a = np.abs(n_1 - n_0)
            dif_b = np.abs(n_2 - n_0)
            if dif_a <= dif_b:
                age2[v[0]] = age1[i] + 1            
                if nodetype2[v[0]] == "null":
                    nodetype2[v[0]] = nodetype1[i]
            else:
                age2[v[1]] = age1[i] + 1            
                if nodetype2[v[1]] == "null":
                    nodetype2[v[1]] = nodetype1[i]
            
    available = np.ones(pos_vertex2.shape[0])
    
    ## If seed is not found => debug
    seed = gt.find_vertex(g2, nodetype2, "Ini")
    if len(seed) == 1:
        seed = seed[0]
    else:
        seed_prev = gt.find_vertex(g1, nodetype1, "Ini")
        p1 = pos1[seed_prev[0]]
        v, d = find_nearest_b(p1, pos_vertex2)
        if d < 20:
            age2[v] = age1[seed_prev[0]] + 1
            nodetype2[v] = "Ini"
            seed = g2.vertex(v)
            available[v] = 0
        else:
            #print('No SEED')
            raise Exception ("No seed point")
    
    pos_vertex2[:,0] = pos_vertex2[:,0] * available
    pos_vertex2[:,1] = pos_vertex2[:,1] * available
    
    ## If main root tip is not found => debug
    end = gt.find_vertex(g2, nodetype2, "FTip")
    if len(end) == 1:
        end = end[0]
    else:
        end_prev = gt.find_vertex(g1, nodetype1, "FTip")
        p1 = pos1[end_prev[0]]
        v, d = find_nearest_b(p1, pos_vertex2)
        if d < 20:
            age2[v] = age1[end_prev[0]] + 1
            nodetype2[v] = "FTip"
            end = g2.vertex(v)
        else:
            v = np.argmax(pos_vertex2[:,1])
            nodetype2[v] = "FTip"
            end = g2.vertex(v)
            # print('No TIP')
            # raise Exception ("BAD TRACKING")
    
    vertices2 = g2.get_vertices()
    for i in vertices2:
        if age2[i] == 0:
            age2[i] == 1
        if nodetype2[i] == "null":
            vecinos = g2.get_out_neighbours(i)
            if len(vecinos) > 1:
                nodetype2[i] = "Bif"
            else:
                if len(vecinos) == 1:
                    nodetype2[i] = "LTip"

    seed = gt.find_vertex(g2, nodetype2, "Ini")
    if len(seed) == 1:
        seed = seed[0]
    else:
        seed_prev = gt.find_vertex(g1, nodetype1, "Ini")
        p1 = pos1[seed_prev[0]]
        v, d = find_nearest_b(p1, pos_vertex2)
        if d < 20:
            age2[v] = age1[seed_prev[0]] + 1
            nodetype2[v] = "Ini"
            seed = g2.vertex(v)
        else:
            #print('No SEED')
            raise Exception ("No seed point")
            
    # edge tracking
    camino, _ = gt.shortest_path(g2, seed, end, weights = weight2)

    l = len(camino)
    for k in range(0, l-1):
        arista = g2.edge(camino[k], camino[k+1])
        clase2[arista][1] = 10

    return [g2, pos2, weight2, clase2, nodetype2, age2]
