#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:38:16 2020

Script to test Multi-coil SOUP Dictionary Learning MRI

@author: Anish Lahiri
"""

import numpy as np
import scipy.io as sio
import dictlearn as dln
import multicoilutils as mcu
#import time
#import matplotlib.pyplot as plt

root = '/mnt/storage/anishl/KneeData_multicoil/'
#root = ''

# load data
fname = 'serie1000481' + 'layer' + '13' + '.mat'
data = sio.loadmat(root+fname)

# load sampling pattern
msk_name = 'Q6_4x' + '.mat'
msk_root = '/home/anishl/SuperUnsuperDictionaryLearning/Masks/'
msk = sio.loadmat(msk_root+msk_name)

# assign k-space, GT, Sensitivity, etc.
k_space = np.moveaxis(np.asarray(data['y']),0,1)
IM0 = np.asarray(data['I1']).T
S = np.asarray(data['S'])
S = np.moveaxis(S,[2,1],[0,1])
Q1 = np.asarray(msk['Q1'].astype('bool')).T



k_space = k_space[:,Q1.flatten()]
IM1 = mcu.A_H(k_space,Q1,S)/Q1.size

PSNR0 = 20*np.log10( np.sqrt(IM1.size)*np.max(np.abs(IM0))/np.linalg.norm(np.abs(IM0)-np.abs(IM1),'fro') )
print('%.2f'%PSNR0,'dB \n')

IM2 = dln.DLrec(IM1, k_space, S, Q1, nu=5e-4,
        lam_max=8.5e-9,
        lam_min=2e-9,
        D0=None,
        patch_shape=(6, 6),
        niter=5,
        ninner=5,
        niter_cg=10,
        stride=1,
        pad='wrap',
        no_pad_axes=(),
        data_update=True)

PSNR_end = 20*np.log10( np.sqrt(IM2.size)*np.max(np.abs(IM0))/np.linalg.norm(np.abs(IM0)-np.abs(IM2),'fro') )
print('%.2f'%PSNR_end,'dB \n')