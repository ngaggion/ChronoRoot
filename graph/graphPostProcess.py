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


def find_dists(i, node, lnodes): #RETURNS THE LIST OF NODES WITHOUT THE NEAREST ONE
    d = np.linalg.norm(node[i]-lnodes, axis = 1)
    
    return d


def trimGraph(grafo, ske, ske2):
    g, pos, weight, clase, nodetype, age = grafo
    
    edges_to_delete = []
    to_delete = []
    
    gt.remove_parallel_edges(g)    
    
    ## DELETE WEIGHT 0 ED
    for edge in g.get_edges():
        e = g.edge(edge[0],edge[1])
        w = weight[e]
        if w == 0:
            edges_to_delete.append(edge)
            v1 = g.get_all_neighbors(edge[0])
            v2 = g.get_all_neighbors(edge[1])
    
            if (len(v1)) == 1:
                to_delete.append(edge[0])
            if (len(v2)) == 1:
                to_delete.append(edge[1])
            
    
    for i in reversed(sorted(to_delete)):
        g.clear_vertex(i)
        g.remove_vertex(i)
        
    to_delete = []
    vertices = g.get_vertices()
    
    for v in vertices:
        vecinos = g.get_out_neighbors(v)
        if len(vecinos) == 2:        
            edge = g.edge(vecinos[0], vecinos[1])
            
            if edge is None:
                edge = g.add_edge(vecinos[0], vecinos[1])
                ed1 = g.edge(v, vecinos[0])
                w1 = weight[ed1]
                ed2 = g.edge(v, vecinos[1])
                w2 = weight[ed2]
    
                weight[edge] = w1+w2
                
                if w1 == 0:
                    clase[edge] = clase[ed2]
                    ske2[np.where(ske2==clase[ed1][0])] == clase[ed2][0]
                elif w2 == 0:
                    clase[edge] = clase[ed1]
                    ske2[np.where(ske2==clase[ed2][0])] == clase[ed1][0]
                else:
                    clase[edge] = clase[ed2]
                    ske2[np.where(ske2==clase[ed1][0])] == clase[ed2][0]
    
                g.remove_edge(ed1)
                g.remove_edge(ed2)
                to_delete.append(v)
    
    for i in reversed(sorted(to_delete)):
        g.clear_vertex(i)
        g.remove_vertex(i)
        
    vertices = g.get_vertices()
    
    pos_vertex = []
    for i in vertices:
        pos_vertex.append(pos[i])
    pos_vertex = np.array(pos_vertex)
    
    pares = []
    
    for i in vertices:
        d = find_dists(i, pos, pos_vertex)
        mask = np.ones(pos_vertex.shape[0], bool)
        mask[i] = False
        
        pair = np.zeros(pos_vertex.shape[0], bool)
        pair[mask] = d[mask]<3
        c = np.count_nonzero(pair)
        
        if c==1:
            k = np.where(pair == True)[0][0]
            if [k, i] not in pares:
                pares.append([i,k])        
    
    to_delete = []
    
    for par in pares:
        v1 = par[0]
        v2 = par[1]
        if g.edge(v1,v2):
            if weight[g.edge(v1,v2)] == 0:
                vecinos2 = g.get_all_neighbors(v2)
                
                for k in vecinos2:
                    if k != v1:
                        edge = g.edge(v2, k)
                        w_e = weight[edge]
                        c_e = clase[edge]
                        
                        n_edge = g.add_edge(v1, k)
                        weight[n_edge] = w_e
                        clase[n_edge] = c_e
                
                g.clear_vertex(v2)
                to_delete.append(v2)
    
    for i in reversed(sorted(to_delete)):
        g.clear_vertex(i)
        g.remove_vertex(i)    
    
    
      
    
    return [g, pos, weight, clase, nodetype, age], ske, ske2
