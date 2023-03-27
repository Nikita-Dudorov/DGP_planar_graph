import scipy as scp
import numpy as np
import matplotlib.pyplot as plt

def Tutte_Embedding(adjacency_list, polygon):
    #calculate Tutte planar embedding of the graph
    #polygon - vertices forming a cycle C s.t G\C is connected, togehter with their predefined coordinates: 
    #list(v,[x,y]), these must form a convex polygon 
    n_vertices=len(adjacency_list)
    data=np.array([])
    row_ind=np.array([],dtype='int64')
    col_ind=np.array([],dtype='int64')
    
    #construct sparse matrix M = Diag(deg(v)) - N, where N[i,j] = (j neighbor of i) ? 1 : 0
    #and M[i,j] = (i = j) ? 1 : 0 for i in predefined polygon
    polygon_vertices=[v for v,coord in polygon]  
    for v, a_list in enumerate(adjacency_list):
        if v in polygon_vertices:
            row_v=np.array([1])
            row_ind_v=np.array([v],dtype='int64')
            col_ind_v=np.array([v],dtype='int64')
        else:
            row_v=-1*np.ones(len(a_list))
            row_v=np.hstack((len(a_list),row_v))
            row_ind_v=v*np.ones(len(a_list)+1,dtype='int64')
            col_ind_v=np.hstack((v,a_list))

        data=np.hstack((data,row_v))
        row_ind=np.hstack((row_ind,row_ind_v))
        col_ind=np.hstack((col_ind,col_ind_v))
        
    M=scp.sparse.csr_matrix((data,(row_ind,col_ind)),shape=(n_vertices, n_vertices))
    b_x=np.zeros(n_vertices)
    b_x[polygon_vertices]=np.array([coord[0] for v,coord in polygon])
    b_y=np.zeros(n_vertices)
    b_y[polygon_vertices]=np.array([coord[1] for v,coord in polygon])
    
    #solve sparse system (Mx=b_x)
    X=scp.sparse.linalg.spsolve(M,b_x)
    Y=scp.sparse.linalg.spsolve(M,b_y)
    return np.transpose(np.vstack((X,Y)))