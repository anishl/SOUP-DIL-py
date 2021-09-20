#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.fft as fft

def A(x,mask,S):
    """
    Implementation of forward System operation in parallel MRI or PFS(.)
    
    In
    x: input image (M x N)
    mask: sampling mask (M x N)
    S: Sensitivity maps (N_c x M x N)
    
    Out
    y_sub : undersampled measuerments (N_c x N_mask)
    """
    xS = x*S
    y_full = fft.fftshift(fft.fft2(fft.ifftshift(xS)))
    y = y_full[:,mask]
    
    return y

def A_H(y,mask,S):
    """
    Implementation of backward System operation in parallel MRI or S*F*P*(.)
    
    In
    y: input undersampled kspace (N_c x N_mask)
    mask: sampling mask (M x N)
    S: Sensitivity maps (N_c x M x N)
    
    Out
    x_zf_sum : sum of zero filled coil images (N_c x M x N)
    """
    y_zf = np.zeros(S.shape,dtype='complex128')
    y_zf[:,mask] = y
    x_zf = fft.fftshift(fft.ifft2(fft.ifftshift(y_zf)))*(mask.size)
    x_zf_sum = ( x_zf*(S.conj()) ).sum(axis=0)
    
    return x_zf_sum

def multicoil_cg(x_in,y,nu,x0,mask,S,n,niter):
    """
     This function uses CG to perform the multi-coil data update on the input image
     using the undersampled measurements, the coil sensitivities and knowedge
     of the samplimg pattern.
                                                                              
     In 
     x_in : output of the Super-Bred algorithm (or x_DL)
     y : measured k-space data from all coils (coil dim = 0)
     S : complex sensitivity maps for all coils (coil_dim = 0)
     mask : undersampling pattern
     x0 : initialization for the updated image
     nu : weight associated with data-fidelity
     niter : number of iterations of CG.
     n : number of considered patches in image x
    
     Out
     x : updated image
    """
    x = x0
    x_zf_sum = A_H(y,mask,S)
#    [:,mask.flatten()]
    
    r = (nu*x_zf_sum) + x_in - (nu*A_H(A(x,mask,S),mask,S)) - (n*x)
    p = r
    rtr_old = np.vdot(r.flatten(),r.flatten())
    
    for i in range(0,niter):
        Ap = (nu*A_H(A(p,mask,S),mask,S)) + (n*p)
        alpha = np.real(rtr_old/np.vdot(p.flatten(),Ap.flatten()))
        
        x += (alpha*p)
        r -= (alpha*Ap)
        
        rtr_new = np.vdot(r.flatten(),r.flatten())
        
        if np.sqrt(rtr_new) < 1e-10:
            break
        
        p = r + (rtr_new*p/rtr_old)
        
        rtr_old = rtr_new
        
    
    return x