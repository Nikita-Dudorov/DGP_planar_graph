import scipy as scp
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from amplpy import AMPL   

#convert a distance matrix to a Gram matrix
def dist2Gram(D):
    n = D.shape[0]
    J = np.eye(n) - (1.0/n)*np.ones((n,n))
    G = -0.5 * J @ (D*D) @ J
    return G

#principal component analysis
def PCA(A,K=2):
    n = A.shape[0]
    evals,evecs = np.linalg.eigh(A)
    evals[evals < 0] = 0  # closest SDP matrix
    sqrootdiag = np.sqrt(np.diag(evals))
    X = evecs @ sqrootdiag
    return X[:,n-K:]

def solve_SDP(D, adjacency_list):
    n = len(adjacency_list)
    G = cp.Variable((n,n), PSD=True) #estimated gram matrix
    
    obj1 = sum([G[i,i]+G[j,j]-2*G[i,j] for i in range(n) for j in adjacency_list[i] if i<j])
    obj2 = cp.trace(G)
    obj = obj1 + 0.01*obj2
    objective = cp.Minimize(obj)

    constraints=[G[i,i]+G[j,j]-2*G[i,j]==D[i,j]**2 for i in range(n) for j in adjacency_list[i] if i<j]
    prob = cp.Problem(objective, constraints)

    #solve the problem
    prob.solve(solver=cp.SCS, verbose=True)
    return G.value

def solve_energy_minimization(adjacency_list, fixed_points):
    n = len(adjacency_list)
    x = cp.Variable(2*n)
    
    obj = sum((x[i]-x[j])**2+(x[i+n]-x[j+n])**2 for i in range(n) for j in adjacency_list[i] if i<j)
    objective = cp.Minimize(obj)

    indices,coordinates=fixed_points
    constraints=[x[i] == p for i,p in zip(indices,coordinates)]
    
    prob = cp.Problem(objective, constraints)

    #solve the problem
    prob.solve()
    return x.value

def solve_DGP_locally(D,adjacency_list,X0):
    nlp = AMPL()
    nlp.read("dgp.mod")
    K=2 #2D-embedding
    n=len(adjacency_list)
    
    #set data
    Kparam = nlp.getParameter('Kdim')
    Kparam.set(K)
    
    nparam = nlp.getParameter('n')
    nparam.set(n)
    
    Eset = nlp.getSet('E')
    data = [(i+1,j+1) for i in range(n) for j in adjacency_list[i] if i<j]       
    Eset.set_values(data)
    
    cparam = nlp.getParameter('c')
    for i in range(n):
        for j in adjacency_list[i]:
            if(i<j):
                cparam.set((i+1,j+1),D[i,j])
                
    # refine solution with a local NLP solver
    nlp.setOption('solver', 'conopt')
    xvar = nlp.getVariable('x')
    for i in range(n):
        for k in range(K):
            xvar[i+1,k+1].setValue(X0[i,k])
    nlp.solve()
    
    X = np.zeros((n,K))
    for i in range(n):
        for k in range(K):
            X[i,k] = xvar[i+1,k+1].value()
    
    return X



def true_distances(X):
    n=X.shape[0]
    distances = np.zeros((n,n))
    for i in range(n):
        distances[i]=np.array([np.linalg.norm(X[i]-X[j]) for j in range(n)])
    return distances

def MDE(X,D,adjacency_list):
    errors=np.array([])
    for v,a_list in enumerate(adjacency_list):
        errors=np.hstack((errors,
        [np.abs(np.linalg.norm(X[v]-X[n])-D[v,n]) for n in a_list if v>n]))
    return np.mean(errors) 

def LDE(X,D,adjacency_list):
    errors=np.array([])
    for v,a_list in enumerate(adjacency_list):
        errors=np.hstack((errors,
        [np.abs(np.linalg.norm(X[v]-X[n])-D[v,n]) for n in a_list if v>n]))
    return np.max(errors) 

def generate_random_points(N,bounds=np.array([[-1,1],[-1,1]])):
    x_bounds,y_bounds=bounds
    points_x = np.random.uniform(x_bounds[0],x_bounds[1],size=N)
    points_y = np.random.uniform(y_bounds[0],y_bounds[1],size=N)
    return np.transpose(np.vstack((points_x,points_y)))  