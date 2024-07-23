import cv2
import matplotlib.pyplot as plt
#done modules locally random
gx, gy, gxlist, gylist = estimate_watermark('./images/fotolia_processed')
cropped_gx, cropped_gy = crop_watermark(gx, gy)
W_m = poisson_reconstruct(cropped_gx, cropped_gy)
img = cv2.imread('images/fotolia_processed/fotolia_137840668.jpg')
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)
num_images = len(gxlist)
J, img_paths = get_cropped_images(
    'images/fotolia_processed', num_images, start, end, cropped_gx.shape)
num_images = min(len(J), 25)
Jt = J[:num_images]
Wm = W_m - W_m.min()
alph_est = estimate_normalized_alpha(Jt, Wm)
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
C, est_Ik = estimate_blend_factor(Jt, Wm, alph)
alpha = alph.copy()
for i in range(3):
    alpha[:, :, i] = C[i] * alpha[:, :, i]
Wm = Wm + alpha * est_Ik
W = Wm.copy()
for i in range(3):
    W[:, :, i] /= C[i]
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
