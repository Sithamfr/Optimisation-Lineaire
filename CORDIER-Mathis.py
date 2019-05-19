

""" PROJET OPTIMISATION LINEAIRE """





import numpy as np






def FM1(A,b):  # A,b de type np.array
    B = np.copy(A)
    c = -np.copy(b)
    c = c.reshape(len(c),1)
    B = np.concatenate((B,c),axis=1)
    return B


def FM2(B):  # B de type np.array 
    C = np.copy(B)
    (p,q) = np.shape(C) 
    for i in range(p): 
        a = abs(C[i,0])
        if a>0:
            C[i,:] = C[i,:]/a
    return C


def FM3(C):  # C a le type np.array
    (p,q) = np.shape(C)
    G,D,E = np.array([np.ones(q)]),np.array([np.ones(q)]),np.array([np.ones(q)])
    for i in range(p):
        x = np.copy(C[i])
        if C[i][0] < .0:
            G = np.concatenate((G,[x]),axis=0)
        elif C[i][0] > .0:
            x=-x
            D = np.concatenate((D,[x]),axis=0)
        else:
            E = np.concatenate((E,[x]),axis=0)  
            
    G = np.delete(G,0,0) # On elimine la premiere ligne.
    D = np.delete(D,0,0)
    E = np.delete(E,0,0)
    
    G = np.delete(G,0,1) # On elimine la premiere colonne 
    D = np.delete(D,0,1)
    E = np.delete(E,0,1)
    
    (pg,qg) = np.shape(G)
    (pd,qd) = np.shape(D)
    if pd == 0:  # si pd=0, D est vide
        return E
    elif pg == 0:  # si pg=0, G est vide
        return E 
    else:
        for k in range(pg):
            for l in range(pd):
                x = G[k,:]-D[l,:]
                E = np.concatenate((E,[x]),axis=0)
    
    return E  



# On calcule ici l'intervalle dans lequel se situe le dernier élément de x
def FM4(A,b):
    E = FM1(A,b)  # B : Concatenation de A et -b
    E = FM2(E)    # C : Première colonne composée de 0,1 et -1
    (p,q) = np.shape(E)
    
    Lx = []
    
    for k in range(q-2):
        E = FM3(E)
        E = FM2(E)
        Lx.append(E)
    
    (p,q) = np.shape(E)
    mini = -np.inf
    maxi = np.inf 
    for k in range(p):
        if E[k,0]<0:
            mini = max(mini, E[k,1])
        elif E[k,0]>0:
            maxi = min(maxi, -E[k,1])
    
    if mini>maxi : raise Exception("Polyèdre vide")
            
    return mini,Lx



# On remonte ici le système en utilisant les matrices E consécutives de FM4
def remontee(L,c):
    mini,vect = L
    X = []
    X0 = mini
    vect.reverse()
    Z = np.array([[1.,0],[0,X0],[0,1]])
    for mat in vect[1:]:
        tmp = np.dot(mat,Z)
        bmax = np.inf
        bmin = -np.inf
        for ligne in tmp:
            if ligne[0]==-1:
                bmin = max(bmin,ligne[1])
            elif ligne[0]==1:
                bmax = min(bmax,-ligne[1])
        X0 = bmin
        Z = np.vstack((Z[0],np.array([0,X0]),Z[1:]))
        X.append(X0)
    X.reverse()
    X = np.hstack(( (mini - np.dot(c[1:],X)) / c[0], X))
    return X




""" FONCTION PRINCIPALE """



def FM(A,b,c):
    # On rajoute un paramètre caractérisant la fonction <c,x> dans A
    # On minimise alors le dernier paramètre
    # Il correspond à la valeur de la fonction
    d = np.concatenate((np.zeros(len(A)),[1,-1]))
    A = np.vstack((A,-c,c))
    A = np.vstack((A.T,d)).T
    b = np.concatenate((b,[0,0]))
    res = FM4(A,b)
    mini_vect = remontee(res,c)
    return (round(res[0],3) , mini_vect)
    




""" TESTS """


        
""" test avec les pillules """
#A=np.array([[3.,2.,1.],
#            [-3.,-2.,-1.],
#            [2.,3.,4.],
#            [-2.,-3.,-4.],
#            [3.,2.,5.],
#            [-3.,-2.,-5.],
#            [-1.,0.,0.],
#            [0.,-1.,0.],
#            [0.,0.,-1.]])
#b=np.array([15.,-6.,12.,-5.,7.,-4.,0.,0.,0.])
#c=np.array([1.,1.,1.])
## ON VEUT X=[1.6,0.6,0], min=2.2




""" autre test """
#A=np.array([[-1.,-1,-1],
#            [-3,1,1],
#            [1,-3,1],
#            [2,1,-3],
#            [0,1,1]])
#b=np.array([1.,1,1,1,4])
#c=np.array([1.,0,-1])
## ON VEUT X=[1,1,3], min=-2
