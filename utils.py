
# std lib
import signal
import threading
import warnings
import os
import difflib
from numbers import Number
from collections.abc import Iterable as _Iterable
from functools import reduce as _reduce
import itertools
import inspect

# 3rd party libs
import pooch
import numpy as np
from numpy.linalg import norm
try:
    from tqdm.auto import tqdm
except ImportError:
    pass


def csign(B):
    """ Complex sign function
    returns exp(1j*B) if B != 0
    otherwise 0.
    np.sign is equivalent to np.sign(B.real)
    """
    if np.isrealobj(B):
        return np.sign(B)
    else:
        return np.exp(1j*np.angle(B))*(B != 0)

def soft_thresholding(B, gamma):
    return ((np.abs(B) >= gamma)*((np.abs(B)-gamma)*csign(B)))

def hard_thresholding(B, gamma):
    return B*(np.abs(B) >= gamma)

def hard_thresholding_inplace(B, gamma):
    # in the real case, we can get away
    # with not allocating memory for abs(B)
    # but in the complex case we have to.
    # mask is allocated in both cases
    # (but two masks are generated in real
    # case, so we only save the diff of bool 
    # over dtype B. Maybe not worth branching over...)
    # todo: there might be np.where options that
    # are more memory efficient... (being a bit pedantic)
    if np.isrealobj(B):
        mask = (B < gamma)
        mask[B <= -gamma] = False
        B[mask] = 0
        return B
    else:
        B[np.abs(B) < gamma] = 0
        return B

# edited from https://stackoverflow.com/a/30110497/5026175
def im2col(A, blk_size, stride=1, pad='constant', no_pad_axes=()):
    """ Image to patch columns

    Extracts overlapping patches with given stride and size
    from the image and vectorizes them.

    Parameters
    ----------
    A: nD array to extract patches from
    blk_size: tuple of patch dimensions
    stride: shift of subsequent extracted patches. Default stride=1
    pad: defines the end conditions. Default pad='constant' which is
        zeros outside of the given image area.
    no_pad_axes: If pad is not None, exclude these axes from padding

    Returns
    -------
    patches: 2D array of patches [prod(blk_size) x ~A.size/prod(strides)]
    """
    if isinstance(stride, Number):
        stride = [stride]*len(blk_size)
    blk_size = [1]*(A.ndim-len(blk_size)) + list(blk_size) 
    if pad is not None:
        blk_pad = [((blk-1)//2, (blk-1)-(blk-1)//2) for blk in blk_size]
        for axis in no_pad_axes:
            blk_pad[axis] = (0,0)
        A = np.pad(A, blk_pad, mode=pad)
    # Parameters
    shp = list(blk_size) + [m-p+1 for m,p in zip(A.shape, blk_size)]
    strd = list(A.strides)*2

    stride = tuple([Ellipsis] + [slice(None, None, s) for s in stride])
    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view[stride].reshape(np.prod(blk_size), -1)


def col2im(patch, shape, blk_size, stride=1, pad='constant', no_pad_axes=(), out=None):
    """ Patch columns to image

    Places patches obtained via im2col back where they came from
    in an image, adding pixels that are represented multiple
    times in the patch matrix.

    Parameters
    ----------
    patches: 2D array of patches [prod(blk_size) x prod(shape)]
    shape: shape of the 2D array to place patches in
    blk_size: tuple of patch dimensions
    stride: shift of subsequent extracted patches. Default stride=1
    pad: defines the end conditions. Default pad='constant' which is
        zeros outside of the given image area.
    no_pad_axes: parameter passed to im2col needed for reconstruction
    out: if provided an array to place output in.

    Returns
    -------
    out: array of collected pixels
    """
    if pad not in ['constant', 'wrap', None]:
        # only ones I think will work right now while using np.roll instead of a true shift
        raise NotImplementedError(
            "Only pad in 'constant', None or 'wrap' are currently supported")
    if pad is None:
        # assert len(no_pad_axes) == 0
        no_pad_axes = [ii for ii in range(len(shape))]
    if isinstance(stride, int):
        stride = [stride]*len(blk_size)
    blk_size = [1]*(len(shape)-len(blk_size)) + list(blk_size)
    stride = [1]*(len(shape)-len(stride)) + list(stride)
    assert len(blk_size) == len(shape)

    if out is None:
        out = np.zeros(shape, dtype=patch.dtype)
    else:
        out.fill(0)

    shape = list(shape)
    for axis in no_pad_axes:
        shape[axis] = shape[axis] - (blk_size[axis]-1)
    shape2 = [int(np.ceil(shp/strd)) for shp,strd in zip(shape,stride)]
    imgs = patch.reshape((*blk_size, *shape2))
    axes = [ii for ii in range(len(shape))]
    slc = [slice(None,None, strd) for strd in stride]
    # the idea here is that each row of a patch column matrix is a shifted version of
    #   the original image. So we reshape each image and de-shift it.
    for indxs in itertools.product(*[range(n) for n in blk_size]): 
        # ^^ equivalent to len(blk_size) nested for loops ^^
        shifts = [(ii-(n-1)//2) for ii, n in zip(indxs,blk_size)]
        for axis in no_pad_axes:
            slc[axis] = slice(indxs[axis],indxs[axis]+shape[axis],stride[axis])
            shifts[axis] = 0
        # print(f'indexs={indxs}, slc={slc}, shifts={shifts}, axes={axes},\n shape={imgs[indxs].shape}, out.shape={out[tuple(slc)].shape}')
        if pad is not None:
            out[tuple(slc)] += np.roll(imgs[indxs], shift=shifts, axis=axes) # undesirable copy...
        else:
            out[tuple(slc)] += imgs[indxs] # we could forego the if, but why waste a copy?
    return out


# This shares a lot of code with the above function, 
# TODO break up into smaller functions that are easier to maintain
def tcol2im(W, patch, shape, blk_size, stride=1, pad='constant', no_pad_axes=(), out=None):
    """ Transform Patch columns to image

    Like col2im(W@patch) but uses memory more efficiently by accumulating
    the result of each row of W multplied by patch into the output.

    Parameters
    ----------
    W: transform matrix (assumed to be square?)
    patches: 2D array of patches [prod(blk_size) x prod(shape)]
    shape: shape of the 2D array to place patches in
    blk_size: tuple of patch dimensions
    stride: shift of subsequent extracted patches. Default stride=1
    pad: defines the end conditions. Default pad='constant' which is
        zeros outside of the given image area.
    no_pad_axes: parameter passed to im2col needed for reconstruction
    out: if provided an array to place output in.

    Returns
    -------
    out: array of collected pixels
    """
    if pad not in ['constant', 'wrap', None]:
        # only ones I think will work right now while using np.roll instead of a true shift
        raise NotImplementedError(
            "Only pad in 'constant', None or 'wrap' are currently supported")
    if pad is None:
        # assert len(no_pad_axes) == 0
        no_pad_axes = [ii for ii in range(len(shape))]
    if isinstance(stride, int):
        stride = [stride]*len(blk_size)
    blk_size = [1]*(len(shape)-len(blk_size)) + list(blk_size)
    stride = [1]*(len(shape)-len(stride)) + list(stride)
    assert len(blk_size) == len(shape)

    if out is None:
        out = np.zeros(shape, dtype=patch.dtype)
    else:
        out.fill(0)

    shape = list(shape)
    for axis in no_pad_axes:
        shape[axis] = shape[axis] - (blk_size[axis]-1)
    shape2 = [int(np.ceil(shp/strd)) for shp,strd in zip(shape,stride)]
    # imgs = patch.reshape((*blk_size, *shape2))
    W2 = W.reshape((*blk_size, -1))
    axes = [ii for ii in range(len(shape))]
    slc = [slice(None,None, strd) for strd in stride]
    # the idea here is that each row of a patch column matrix is a shifted version of
    #   the original image. So we reshape each image and de-shift it.
    for indxs in itertools.product(*[range(n) for n in blk_size]): 
        # ^^ equivalent to len(blk_size) nested for loops ^^
        shifts = [(ii-(n-1)//2) for ii, n in zip(indxs,blk_size)]
        for axis in no_pad_axes:
            slc[axis] = slice(indxs[axis],indxs[axis]+shape[axis],stride[axis])
            shifts[axis] = 0
        # print(f'indexs={indxs}, slc={slc}, shifts={shifts}, axes={axes},\n shape={imgs[indxs].shape}, out.shape={out[tuple(slc)].shape}')
        img = (W2[indxs]@patch).reshape(shape2)
        if pad is not None:
            out[tuple(slc)] += np.roll(img, shift=shifts, axis=axes) # undesirable copy...
        else:
            out[tuple(slc)] += img # we could forego the if, but why waste a copy?
    return out



def im2col_weights(shape, *args, **kwargs):
    """ im2col Weights

    Weight matrix representing how many times each pixel in an image
    is represented in a patch matrix obtained via im2col.
    """
    patches = im2col(np.ones(shape), *args, **kwargs)
    weights = col2im(patches, shape, *args, **kwargs)
    return weights


# edited from https://stackoverflow.com/a/30110497/5026175
def im2col_old(A, blk_size, stride=1, pad='constant'):

    if pad is not None:
        blk_pad = [((blk-1)//2, (blk-1)-(blk-1)//2) for blk in blk_size]
        A = np.pad(A, blk_pad, mode=pad)
    # Parameters
    m, n = A.shape
    s0, s1 = A.strides
    nrows = m-blk_size[0]+1
    ncols = n-blk_size[1]+1
    shp = blk_size[0], blk_size[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(blk_size[0]*blk_size[1], -1)[:, ::stride]


def col2im_old(patch, shape, blk_size, stride=1, pad='constant'):

    # things I didn't implement because I'm lazy...
    if stride != 1:
        # this should take two seconds if I think about it, requires adding to specific indexes
        raise NotImplementedError("stride != 1 not currently supported")
    if pad not in ['constant', 'wrap']:
        # only ones I think will work right now while using np.roll instead of a true shift
        raise NotImplementedError(
            "Only pad in 'constant' or 'wrap' are currently supported")
    if isinstance(stride, Number):
        stride = [stride]*len(blk_size)
    blk_size = [1]*(len(shape)-len(blk_size)) + list(blk_size) 

    # ims = []
    A = np.zeros(shape, dtype=patch.dtype)

    # the idea here is that each row of a patch column matrix is a shifted version of
    #   the original image. So we reshape each image and de-shift it.
    m, n = blk_size
    cm, cn = m//2, n//2
    ncm, ncn = -m//2, -n//2
    for ii in range(ncm+1, cm+1):
        for jj in range(ncn+1, cn+1):
            kk = n*(ii-ncm-1) + (jj-ncn-1)  # should be equivalent to k = k + 1
            A += (np.roll(patch[kk].reshape(shape),
                          shift=(ii, jj), axis=(0, 1)))
    assert kk == patch.shape[0]-1  # did we visit every column?
    return A


def dctmtx(N):
    """Creates an NxN matrix that performs the
    N-point DCT transform on a vector"""
    n = np.arange(0, N)
    k = n
    dctm = np.cos(np.pi*np.outer(k, (n+0.5))/(N))
    dctm *= np.sqrt(2/N)
    dctm[0, :] /= np.sqrt(2)
    return dctm


def dct2dmtx(N, M=None):
    """Creates a NM x NM matrix that performs
    the (N,M)-point DCT in two dimensions on a vectorized
    image."""
    if M is None:
        if isinstance(N, tuple):
            N, M = N
        else:
            M = N
    return np.kron(dctmtx(N), dctmtx(M))


def _isiterable(obj):
    if isinstance(obj, _Iterable):
        return True
    elif hasattr(obj, "__getitem__") and hasattr(obj, "__len__"):
        return True
    else:
        return False


def _accept_tuple_or_args(args):
    if len(args) == 1:
        if _isiterable(args[0]):
            return args[0]
        else:
            return (args,)
    else:
        return args


def nd_dctmtx(*args):
    """Creates a square Nd - DCT Matrix"""
    args = _accept_tuple_or_args(args)
    return _reduce(np.kron, (dctmtx(arg) for arg in args))


def overcomplete_dctmtx(K, n):
    bbb = n
    Pn = K
    DCT = np.zeros((bbb, Pn))
    for k in range(Pn):
        V = np.cos(np.arange(bbb).T*k*np.pi/Pn)
        if k > 0:
            V = V-np.mean(V)
        DCT[:, k] = V/norm(V)
    return DCT.T


def overcomplete_dct2dmtx(K, n):
    """ Overcomplete 2D DCT Matrix

    if n is a tuple, then it is a 
    n[0] x n[1] 2D DCT, otherwise
    it is a sqrt(n) x sqrt(n) 2D DCT
    """
    if isinstance(n, tuple):
        patch_shape = n
        n = np.prod(patch_shape)
    else:
        patch_shape = [int(np.sqrt(n))]*2
    if K <= n:
        W0 = dct2dmtx(*patch_shape)
        W0 = W0[0:K]
    elif K > n:
        DCT1 = overcomplete_dctmtx(int(np.ceil(np.sqrt(K))), patch_shape[0])
        DCT2 = overcomplete_dctmtx(int(np.ceil(np.sqrt(K))), patch_shape[1])
        # I am not confident in this order...
        DCT = np.kron(DCT1, DCT2)
        W0 = DCT[0:K]
    return W0


def mse(estim, truth):
    """Calculates the Mean Square Error"""
    return (np.abs((truth-estim))**2).mean()


def psnr(estim, truth):
    """Calculates the Peak-Signal-to-Noise Ratio in dB"""
    true_max = np.abs(truth).max()**2
    return 10*np.log10(true_max/mse(estim, truth))


def irange(start, stop=None, step=1, *,
           verbose=True,
           delay_interrupt=False,
           pbar=False,
           **kwargs):
    """ irange
    !!! NOT THREAD SAFE !!! Only use for interactive main thread uses
    An improved range function that provides added functionality such as
        - tqdm support for progress bars
        - Delayed or Suppressed Interrupt Handling till end of iteration

    Options
    -------
    delay_interrupt if True, catch up to 1 Keyboard Interrupts and delay 
                    to end of one iteration. 
                    if 'raise', the interrupt is raised at end of iteration
                    otherwise it is suppressed.

    pbar            display progress bar suitable for the environment
    **kwargs        kwargs for tqdm
                    

    Note: This function sets custom handlers for interrupts and then
        optionally drops them. This may have uninteded consequences.
        Users Beware.

    """
    if pbar is None:
        try:
            tqdm
            pbar = True
        except NameError:
            pbar = False
    if stop is None:
        stop = start
        start = 0

    if delay_interrupt:
        if threading.current_thread() is not threading.main_thread():
            warnings.warn("Using irange to delay interrupts is not recommended outside of the main thread")
        old_handler = signal.getsignal(signal.SIGINT)

        class int_handler(object):
            def __init__(self):
                self.signal_received = False

            def handler(self, sig, frame, old_handler=old_handler):
                if not self.signal_received:
                    self.signal_received = (sig, frame)
                    if verbose: print("Interrupt Received...")
                else:  # raise if received a second interrupt
                    signal.signal(signal.SIGINT, old_handler)
                    if old_handler:
                        old_handler(*self.signal_received)
        ih = int_handler()
        old_hand2 = signal.signal(signal.SIGINT, ih.handler)
        assert old_handler is old_hand2
    else:
        ih = None

    if pbar and (stop-start) > 1:
        r = tqdm(range(start, stop, step), **kwargs)
    else:
        r = range(start, stop, step)
        for kwarg in ['desc', 'total', 'ncols']:
            kwargs.pop(kwarg,None)
        if len(kwargs) != 0:
            raise ValueError("Unused key word arguments in irange")

    for ii in r:
        yield ii
        if ih is not None and ih.signal_received:
            if verbose: print("Iteration Finished")
            signal.signal(signal.SIGINT, old_handler)
            if delay_interrupt == 'raise' and old_handler:
                old_handler(*ih.signal_received)
            break
    try:
        signal.signal(signal.SIGINT, old_handler)
    except NameError: # i.e. old_handler not defined
        pass


def rand_rect(shape, num=7):
    """Random Sum of Rectangles

    Generates a sum of `num` rectangles
    of random width.

    Parameters
    ----------
    shape: shape of generated signal, if int, returns a 1D
            signal, if 2-tuple, returns an image
    num: number of rectangles to add together

    """
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(shape, tuple):
        if len(shape) == 1:
            M, N = shape[0], 1
        else:
            M, N = shape
    l, w = M//3, N//3
    img = np.zeros((M, N))
    for ii in range(num):
        tl_y = np.random.randint(M - 1.5*l) if M > 1 else 0
        tl_x = np.random.randint(N - 1.5*w) if N > 1 else 0
        br_y = np.random.randint(tl_y + l, M) if M > 1 else 1
        br_x = np.random.randint(tl_x + w, N) if N > 1 else 1
        img[tl_y:br_y, tl_x:br_x] += 1
    return img/img.max() if len(shape) > 1 else np.squeeze(img/img.max())


def batch_rand_rect(shape, num=7):
    """ Returns a generator that yields shape[0] examples
    one-by-one. If you would like all of the examples then
    just pass it to a list or array like so:

    >>> all_train = list(batch_rand_rect((500,32,32)))
    >>> all_train = np.array(list(batch_rand_rect((500,32,32))))
    or process this one by one:
    >>> for example in batch_rand_rect((500,32,32)):
    >>>    assert example.shape == (32,32)
    >>> img_gen = batch_rand_rect((500,32,32))
    >>> example1 = next(img_gen)
    >>> example2 = next(img_gen)

    This allows us to process a lot of examples without
    keeping them all in memory.
    """
    batch_size = shape[0]
    shape = shape[1:]
    return (rand_rect(shape, num) for _ in range(batch_size))


def rand_mod_pulse(shape, num=3):
    """Sum of Random Modulated Pulses

    Generates a sum of `num` modulated gaussian pulses of
    random width and random frequency.

    Parameters
    ----------
    shape: shape of generated signal, if int, returns a 1D
            signal, if 2-tuple, returns an image
    num: number of pulses to add together

    """
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(shape, tuple):
        if len(shape) == 1:
            M, N = shape[0], 1
        else:
            M, N = shape
    X, Y = np.meshgrid(np.r_[0:N], np.r_[0:M])
    img = np.zeros((M, N))
    for ii in range(num):
        rx = np.random.randint(N-1)+1 if N > 1 else 1
        ry = np.random.randint(M-1)+1 if M > 1 else 1
        cx = np.random.randint(rx//2, N-rx//2)
        cy = np.random.randint(ry//2, M-ry//2)
        phi = np.random.rand()*2*np.pi
        wx = np.random.rand()*N/3
        wy = np.random.rand()*M/3
        img += np.exp(-((X-cx)**2/(15*rx) + (Y-cy)**2/(15*ry)))*(1+np.cos((wx*X/N+wy*Y/M) + phi))
    return img/img.max() if len(shape) > 1 else np.squeeze(img/img.max())
