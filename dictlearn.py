#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import utils
import multicoilutils as mcu



def SOUPDILLO(Y,D0,X0,l2,numiter):
    """
    Function to learn a dictionary (D) and corresponding sparse codes (Z) to represent a set of signals (Y)
    using the SOUP-DIL method
    
    In
        Y: data matrix with vectorized image patches along its columns (Npix x Npatches)
        D0: initial Dictionary
        X0: inital sparse codes
        l2: sparsity penalty
        numiter: number of iterations for sparse coding and dictionary learning
        
    Out
        D: learned dictionary
        C.conj().T: sparse codes
        D@C.conj().T: dictionary Learned sparse representation
    """
    D = D0 
    C = X0.conj().T
    
    n = D.shape[0]
    K = D.shape[1]
    In = np.identity(n); v1 = In[:,0];
    
    for i in range(0,numiter):
        
        for j in range(0,K):
            bt = (Y.conj().T@D[:,j]) - (C@D.conj().T@D[:,j]) + C[:,j]
            
            cjt =  bt*(np.abs(bt) >= l2)
            
            ht = (Y@cjt) - (D@(C.conj().T@cjt)) + D[:,j]*np.vdot(C[:,j],cjt);
            
            if np.any(cjt):
                djt = ht/np.linalg.norm(ht,2)
            else:
                djt = v1
                    
        C[:,j] = cjt
        D[:,j] = djt
        
    return D,C.conj().T, D@C.conj().T
    
def SOUPDILLO_fast(Y,D0,X0,l2,numiter):
    """
    INCOMPLETE!!!!! WORK IN PROGRESS
    
    
    
    More efficient implementation of a function to learn a dictionary (D) and corresponding sparse codes (Z) to represent a set of signals (Y)
    using the SOUP-DIL method
    
    In
        Y: data matrix with vectorized image patches along its columns (Npix x Npatches)
        D0: initial Dictionary
        X0: inital sparse codes
        l2: sparsity penalty
        numiter: number of iterations for sparse coding and dictionary learning
        
    Out
        D: learned dictionary
        C.conj().T: sparse codes
        D@C.conj().T: dictionary Learned sparse representation
    """
    D = D0
    X = X0
    
    n,K  = D.shape
    In = np.identity(n); v1 = In[:,0];
    sd = range(K)
    
    for i in range(numiter):
        ZP = D.conj().T@Y
        rowx,colx = np.where(X)
        for j in sd:
            ind_r = rowx==j
            ind_c= colx[ind_r]
            gamm = X[ind_r,ind_c]
            Dj = D[:,j]
            
            h = -(Dj.conj.T@D)@X
            h[ind_c] = h[ind_c] + gamm
            h = ZP[j,:] + h
            
            h = h*(np.abs(h)>=l2)
            ix = np.where(h)
            if not np.any(h):
                Dj = v1
            else:
                Dj  
            

def DLrec(IM, k_space, S, mask, nu,
        lam_max=0.2,
        lam_min=0.05,
        D0=None,
        patch_shape=(6, 6),
        niter=10,
        ninner=5,
        niter_cg=5,
        stride=1,
        pad='wrap',
        no_pad_axes=(),
        data_update=False,
        **kwargs):
    """
    Function to perform dictionary learning reconstrucion of MRI Data
    
    In
        IM: initial image to start the reconstruction with (usually zero filled sum of coils)
        k_space: undersampled k-space measurements (coil dim = 0)
        S: coil sensitivity maps (coil dim = 0)
        mask: undersampling mask used for acquisition
        nu: weight on data fidelity term
        lam_max: maximum sparsity penalty
        lam_min: minimum sparseity penalty
        D0: initial Dictionary
        patch_shape: size of patches extracted
        niter: number of outer iterations of DL+CG
        ninner: number of SOUP-DIL iterations for learning the dictionary
        niter_cg: number of iterations of conjugate gradient descent
        stride: stride during patch extraction (must be 1)
        pad: type of padding for the edges of the image
        data_update: True indicates data update is necessary on the DL image output
        
    Out
        IM2: the output of SOUP-DLMRI with or without data update (as specified)
        
    """
    if D0 is None:
        n = patch_shape[0]*patch_shape[1]
        D0 = utils.overcomplete_dct2dmtx(n, 4*n).astype('complex128')
        
    IM2 = IM
    l2s = np.logspace(np.log10(lam_max),np.log10(lam_min),niter)
    
    for it in range(0,niter):
        Y = utils.im2col(IM2, patch_shape, stride=stride, pad=pad, no_pad_axes=no_pad_axes)
        X0 = np.zeros((D0.shape[1],Y.shape[1])).astype('complex128')

        _,_,Y_approx = SOUPDILLO(Y,D0,X0,l2s[it],ninner)

        IM = utils.col2im(Y_approx, IM.shape, patch_shape,
                          stride=stride, pad=pad, no_pad_axes=no_pad_axes,
                          out=None)
                
        if data_update:
            IM2 = mcu.multicoil_cg(IM,k_space,nu,IM2,mask,S,n,niter_cg)
        else:
            IM2 = IM
        
    return IM2
