import scipy as scp
import numpy as np
import matplotlib.pyplot as plt

def matrix_to_list(M):
    #transform matrix represenation M of the graph into adjacency list representation
    return [np.squeeze((row>0).nonzero()) for row in M]

def list_to_sparseMatrix(adjacency_list):
    #transform adjacency list representation of the graph into sparse matrix representation 
    n = len(adjacency_list)
    data=np.array([])
    row_ind=np.array([],dtype='int64')
    col_ind=np.array([],dtype='int64')
    
    for v, a_list in enumerate(adjacency_list):
        row_v=np.ones(len(a_list))
        row_ind_v=a_list
        col_ind_v=v*np.ones(len(a_list),dtype='int64')
        
        data=np.hstack((data,row_v))
        row_ind=np.hstack((row_ind,row_ind_v))
        col_ind=np.hstack((col_ind,col_ind_v))
        
    M = scp.sparse.csr_matrix((data,(row_ind,col_ind)),shape=(n, n))
    return M

def count_edges(adjacency_list):
    return sum([len(a_list) for a_list in adjacency_list])/2

def lists_to_sets(adjacency_list):
    graph = []
    for i,a_list in enumerate(adjacency_list):
        graph.append(set(a_list))
    return graph

def dfs(graph, start, visited=None):    
    if visited is None:
        visited = set()
    visited.add(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited

def graph_to_distances(X, adjacency_list):
    #calculcate matrix of pairwise distacnes between vertices of the graph
    #X - list of [x,y] coordinates of vertices
    n_vertices = X.shape[0]
    data=np.array([])
    row_ind=np.array([],dtype='int64')
    col_ind=np.array([],dtype='int64')
    
    for v,a_list in enumerate(adjacency_list):
        row_v=[np.linalg.norm(X[v]-X[n]) for n in a_list]
        row_ind_v=v*np.ones(len(a_list),dtype='int64')
        col_ind_v=a_list

        data=np.hstack((data,row_v))
        row_ind=np.hstack((row_ind,row_ind_v))
        col_ind=np.hstack((col_ind,col_ind_v))

    D=scp.sparse.csr_matrix((data,(row_ind,col_ind)),shape=(n_vertices, n_vertices))
    return D

def triangulation_to_graph(n,triangulation):
    #n - number of vertices
    #triangulation - list of faces [v1,v2,v3]
    adjacency_list = [np.array([],dtype='int64') for i in range(n)]
    for triangle in triangulation:
        v1,v2,v3 = triangle
        adjacency_list[v1]=np.hstack((adjacency_list[v1],[v2,v3]))
        adjacency_list[v2]=np.hstack((adjacency_list[v2],[v1,v3]))
        adjacency_list[v3]=np.hstack((adjacency_list[v3],[v1,v2]))
    for v in range(n):
        adjacency_list[v] = np.array(list(set(adjacency_list[v]))) #remove repeated nieghbors
    return adjacency_list     

def randomFace_of_triangulation(triangulation):
    #return list of vertices forming the face togehter with their coordinates: 
    #list(v,[x,y]) 
    ind = np.random.randint(len(triangulation.simplices))
    face = [(v,p) for v,p in zip(triangulation.simplices[ind],
                                 triangulation.points[triangulation.simplices[ind]])] 
    return face 

def boundary_of_triangulation(triangulation):
    #return list of vertices forming the boundary face togehter with their coordinates: 
    #list(v,[x,y])
    three_connected=True
    boundary_vertices=np.array([],dtype='int64')
    for neighbor_faces,face in zip(triangulation.neighbors,triangulation.simplices):
        n_boundary_edges = len(neighbor_faces[neighbor_faces==-1])
        if(n_boundary_edges==0):
            continue
        elif(n_boundary_edges==1):
            boundary_vertices = np.hstack((boundary_vertices,face[neighbor_faces!=-1]))
        else:
            boundary_vertices = np.hstack((boundary_vertices,face))
            three_connected = False
    boundary_vertices=np.array(list(set(boundary_vertices))) #remove double vertices
    boundary = [(v,p) for v,p in zip(boundary_vertices,triangulation.points[boundary_vertices])]
    return boundary, three_connected

def randomly_remove_edges(n_edges2remove, adjacency_list):
    n_edges=count_edges(adjacency_list)
    n_vertices=len(adjacency_list)
    if((n_edges-n_edges2remove)<(n_vertices-1)):
        raise Exception("tried to make appear isolated vertices")
        
    reduced_list = adjacency_list.copy()
    removed=0
    while(removed<n_edges2remove):
        v=np.random.randint(low=0,high=n_vertices)
        v_neighbors=len(reduced_list[v])
        n_ind=np.random.randint(low=0,high=v_neighbors)
        n=reduced_list[v][n_ind]
        v_ind=np.where(reduced_list[n]==v)
        n_neighbors=len(reduced_list[n])
        if((v_neighbors<=1) | (n_neighbors<=1)): #avoid isolated vertices 
            continue
        reduced_list[v]=np.delete(reduced_list[v],n_ind)
        reduced_list[n]=np.delete(reduced_list[n],v_ind)
        removed+=1
        
    visited_vertices=dfs(lists_to_sets(reduced_list),start=0)
    if(len(visited_vertices)<n_vertices):
        connected=False
    else:
        connected=True
        
    return reduced_list,connected





def draw_graph(X, adjacency_list, draw_points=True, edge_width=1):
    #X - list of [x,y] coordinates of vertices
    if(draw_points):
        plt.scatter(X[:,0],X[:,1],color='red')
    for v, a_list in enumerate(adjacency_list):
        [plt.plot([X[v,0],X[n,0]],[X[v,1],X[n,1]],color='red',linewidth=edge_width) for n in a_list]
            
def draw_triangulation(points, triangulation, draw_points=True, edge_width=1):
    #points - list of [x,y] coordinates of vertices
    #triangulation - list of faces [v1,v2,v3]
    if(draw_points):
        plt.scatter(points[:,0], points[:,1],color='red')
    plt.triplot(points[:,0], points[:,1],triangulation,color='red',linewidth=edge_width)