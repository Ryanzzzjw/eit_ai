import numpy as np





if __name__ == "__main__":

    n_samples=1000
    m_meas= 256
    n_elem= 3000
    X=np.array([[i+j for i in range(m_meas)] for j in range(n_samples)])

    Y=np.array([[i+j*0.1 for i in range(n_elem)] for j in range(n_samples)])
    
    xyz= np.array([[i,i,i] for i in range(n_elem)])

    print(X, X.shape)#
    print(Y, Y.shape)
    print(xyz, xyz.shape)
    # x= np.array([[] for _ in range(n_samples+3)])

    for idx_samples in range(n_samples):
        X_s= np.array([X[idx_samples] for _ in range(n_elem)])

        tmp =np.concatenate((X_s, xyz), axis=1)
        if idx_samples==0:
            x=tmp
        else:
            x= np.concatenate((x, tmp), axis=0)



    # x=np.zeros((n_samples*n_elem, m_meas))
    print(x, x.shape)

    y= Y.flatten()
    print(y, y.shape)