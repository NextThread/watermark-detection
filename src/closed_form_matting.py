from __future__ import division

def rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)

def computeLaplacian(img, eps=10**(-7), win_rad=1):
    win_size = (win_rad*2+1)**2
    h, w, d = img.shape
    c_h, c_w = h - 2*win_rad, w - 2*win_rad
    win_diam = win_rad*2+1

    indsM = np.arange(h*w).reshape((h, w))
    ravelImg = img.reshape(h*w, d)
    win_inds = rolling_block(indsM, block=(win_diam, win_diam))
    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=2, keepdims=True)
    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    return L


def closed_form_matte(img, scribbled_img, mylambda=100):
    h, w,c  = img.shape
    consts_map = (np.sum(abs(img - scribbled_img), axis=-1)>0.001).astype(np.float64)
    consts_vals = scribbled_img[:,:,0]*consts_map
    D_s = consts_map.ravel()
    b_s = consts_vals.ravel()
    L = computeLaplacian(img)
    sD_s = scipy.sparse.diags(D_s)
    x = scipy.sparse.linalg.spsolve(L + mylambda*sD_s, mylambda*b_s)
    alpha = np.minimum(np.maximum(x.reshape(h, w), 0), 1)
    return alpha
