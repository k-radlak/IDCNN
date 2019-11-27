# Generated with SMOP  0.41
#from smop.libsmop import *
# median_reconstruction.m
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image

from skimage import data
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage.filters import laplace
import skimage
from skimage.measure import label, regionprops
import cv2

def opencv_reconstruction(img,mask,method):
    img = np.asarray(img)
    mask = np.asarray(mask)
    inpaint_mask = mask.astype("uint8")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if method==0:
        img_out = cv2.inpaint(img, inpaint_mask, 3, cv2.INPAINT_TELEA)
    elif method==1:
        img_out = cv2.inpaint(img, inpaint_mask, 3, cv2.INPAINT_NS)
    else:
        assert "Unknown method"

    return img_out


def median_reconstruction(img=None,mask=None,threshold=None):
    img = np.asarray(img)
    mask = np.asarray(mask)
    imgOut=np.copy(img)
    
    x = img.shape[0]
    y = img.shape[1]
    for i in range(0,x):
        for j in range(0,y):
            if (mask[i,j] > 0):
                param=1
                while True:

                    xlow=max(i - param,0)
                    ylow=max(j - param,0)
                    xhigh=min(i + param+1,x)
                    yhigh=min(j + param+1,y)
                    xx=0
                    yy=0
                    submask=np.logical_not(mask[xlow:xhigh,ylow:yhigh])
                    if np.sum((submask)) >= threshold:
                        break
                    else:
                        param=param + 1
                xlow=max(i - param,0)
                ylow=max(j - param,0)
                xhigh=min(i + param+1,x)
                yhigh=min(j + param+1,y)
                pixels=np.empty((0,3), int)
                pixels_id=np.empty((0,2), int)
                for a in range(xlow,xhigh):
                    for b in range(ylow,yhigh):
                        if mask[a,b] == 0:
                            tmp=np.reshape(img[a,b,0:3],(1,3))
                            pixels=np.append(pixels,tmp,axis=0)
                            t2= np.array([a,b])
                            pixels_id=np.vstack([pixels_id,t2])
                dist=np.empty((0,1), float)
                for t in range(pixels.shape[0]):
                    out = np.linalg.norm(pixels- pixels[t,:], axis=1 )#cdist(pixels, pixels[t,:], 'sqeuclidean')
                    dist=np.append(dist,np.sum(out))
                id=np.argmin(out)
                imgOut[i,j,:]=img[pixels_id[id,0],pixels_id[id,1],:]
    return imgOut # Image.fromarray(imgOut, 'RGB')


def mean_reconstruction(img=None, mask=None, threshold=None):
    img = np.asarray(img)
    mask = np.asarray(mask)
    imgOut = np.copy(img)

    x = img.shape[0]
    y = img.shape[1]
    for i in range(0, x):
        for j in range(0, y):
            if (mask[i, j] >0):
                param = 1
                while True:

                    xlow = max(i - param, 0)
                    ylow = max(j - param, 0)
                    xhigh = min(i + param + 1, x)
                    yhigh = min(j + param + 1, y)
                    xx = 0
                    yy = 0
                    submask = np.logical_not(mask[xlow:xhigh, ylow:yhigh])
                    if np.sum((submask)) >= threshold:
                        break
                    else:
                        param = param + 1
                xlow = max(i - param, 0)
                ylow = max(j - param, 0)
                xhigh = min(i + param + 1, x)
                yhigh = min(j + param + 1, y)
                pixels = np.empty((0, 3), int)
                for a in range(xlow, xhigh):
                    for b in range(ylow, yhigh):
                        if mask[a, b] == 0:
                            tmp = np.reshape(img[a, b, 0:3], (1, 3))
                            pixels = np.append(pixels, tmp, axis=0)

                tmp_out=np.mean(pixels,axis=0)

                imgOut[i, j, :] = np.reshape(tmp_out,(1,1,3))
    return imgOut#Image.fromarray(imgOut, 'RGB')

def numbergrid(mask):
    n = np.sum(mask)
    G1 = np.zeros_like(mask, dtype=np.uint32)
    G1[mask] = np.arange(1, n+1)
    return G1

def delsq_laplacian(G):
    [m, n] = G.shape
    G1 = G.flatten()
    p = np.where(G1)[0]
    N = len(p)
    i = G1[p] - 1
    j = G1[p] - 1
    s = 4 * np.ones(N)
    #for all four neighbor of center points
    for offset in [-1, m, 1, -m]:
        #indices of all possible neighbours in sparse matrix
        Q = G1[p+offset]
        #filter inner indices
        q = np.where(Q)[0]
        #generate neighbour coordinate
        i = np.concatenate([i, G1[p[q]]-1])
        j = np.concatenate([j, Q[q]-1])
        s = np.concatenate([s, -np.ones(q.shape)])

    sp = sparse.csr_matrix((s, (i,j)), (N,N))
    return sp

def delsq_bilaplacian(G):
    [n, m] = G.shape
    G1 = G.flatten()
    p = np.where(G1)[0]
    N = len(p)
    i = G1[p] - 1
    j = G1[p] - 1
    s = 20 * np.ones(N)
    #for all four neighbor of center points
    coeffs  = np.array([1, 2, -8, 2, 1, -8, -8, 1, 2, -8, 2, 1])
    offsets = np.array([-2*m, -m-1, -m, -m+1, -2, -1, 1, 2, m-1, m, m+1, 2*m])
    for coeff, offset in zip(coeffs, offsets):
        #indices of all possible neighbours in sparse matrix
        Q = G1[p+offset]
        #filter inner indices
        q = np.where(Q)[0]
        #generate neighbour coordinate
        i = np.concatenate([i, G1[p[q]]-1])
        j = np.concatenate([j, Q[q]-1])
        s = np.concatenate([s, coeff*np.ones(q.shape)])

    sp = sparse.csr_matrix((s, (i,j)), (N,N))
    return sp

def generate_stencials():
    stencils = []
    for i in range(5):
        for j in range(5):
            A = np.zeros((5, 5))
            A[i,j]=1
            S = laplace(laplace(A))
            x_range = np.array([i-2, i+3]).clip(0,5)
            y_range = np.array([j-2, j+3]).clip(0,5)
            S = S[x_range[0]:x_range[1], y_range[0]:y_range[1]]
            stencils.append(S)

    return stencils

def _inpaint_biharmonic_single_channel(mask, out, limits):
    # Initialize sparse matrices
    matrix_unknown = sparse.lil_matrix((np.sum(mask), out.size))
    matrix_known = sparse.lil_matrix((np.sum(mask), out.size))

    # Find indexes of masked points in flatten array
    mask_i = np.ravel_multi_index(np.where(mask), mask.shape)

    G = numbergrid(mask)
    L = delsq_bilaplacian(G)
    out[mask] = 0
    B = -laplace(laplace(out))
    #plt.imshow(B, cmap='gray')
    #plt.show()
    b = B[mask]
    result = spsolve(L, b)
    # Handle enormous values
    result = np.clip(result, *limits)
    result = result.ravel()
    out[mask] = result
    return out

def dilate_rect(rect, d, nd_shape):
    rect[0:2] = (rect[0:2] - d).clip(min = 0)
    rect[2:4] = (rect[2:4] + d).clip(max = nd_shape)
    return rect

def k_inpaint_biharmonic(image, mask, multichannel=False):
    if image.ndim < 1:
        raise ValueError('Input array has to be at least 1D')

    img_baseshape = image.shape[:-1] if multichannel else image.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')

    if np.ma.isMaskedArray(image):
        raise TypeError('Masked arrays are not supported')

    image = skimage.img_as_float(image)
    mask = mask.astype(np.bool)

    # Split inpainting mask into independent regions
    kernel = ndi.morphology.generate_binary_structure(mask.ndim, 1)
    mask_dilated = ndi.morphology.binary_dilation(mask, structure=kernel)
    mask_labeled, num_labels = label(mask_dilated, return_num=True)
    mask_labeled *= mask
    if not multichannel:
        image = image[..., np.newaxis]

    out = np.copy(image)

    props = regionprops(mask_labeled)
    comp_out_imgs = []
    comp_masks = []
    for i in range(num_labels):
        rect = np.array(props[i].bbox)
        rect = dilate_rect(rect, 2, image.shape[:2])
        out_sub_img = out[rect[0]:rect[2], rect[1]:rect[3], :]
        comp_mask   = mask[rect[0]:rect[2], rect[1]:rect[3]]
        # plt.subplot(121), plt.imshow(comp_mask)
        # plt.subplot(122), plt.imshow(out_sub_img)
        # plt.show()
        comp_out_imgs.append(out_sub_img)
        comp_masks.append(comp_mask)

    for idx_channel in range(image.shape[-1]):
        known_points = image[..., idx_channel][~mask]
        limits = (np.min(known_points), np.max(known_points))
        for i in range(num_labels):
            _inpaint_biharmonic_single_channel(comp_masks[i], comp_out_imgs[i][..., idx_channel], limits)

    if not multichannel:
        out = out[..., 0]

    return out