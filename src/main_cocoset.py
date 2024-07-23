

IMAGE_FOLDER = "/media/rohitrango/2EC8DBB2C8DB7715/"
IMG_LOC = "coco_dataset"
IMG_PROCESSED_LOC = "coco_dataset_processed"

def get_alpha_matte(watermark, threshold=128):
	w = np.average(watermark, axis=2)
	_, w = cv2.threshold(w, threshold, 255, cv2.THRESH_BINARY_INV)
	return PlotImage(w)

def P(img,e=None):
    if e is None:
        plt.imshow(PlotImage(img)); plt.show()
    else:
        plt.imshow(PlotImage(img),'gray'); plt.show()

def bgr2rgb(img):
	return img[:,:,[2, 1, 0]]

if __name__ == "__main__":
	foldername = os.path.join(IMAGE_FOLDER, IMG_PROCESSED_LOC)
	gx, gy, gxlist, gylist = estimate_watermark(foldername)

	cropped_gx, cropped_gy = crop_watermark(gx, gy)
	W_m = poisson_reconstruct(cropped_gx, cropped_gy, num_iters=5000)

	img = cv2.imread(os.path.join(foldername, '000000051008.jpg'))
	im, start, end = watermark_detector(img, cropped_gx, cropped_gy)
	num_images = len(gxlist)

	J, img_paths = get_cropped_images(foldername, num_images, start, end, cropped_gx.shape)
	idx = [389, 144, 147, 468, 423, 92, 3, 354, 196, 53, 470, 445, 314, 349, 105, 366, 56, 168, 351, 15, 465, 368, 90, 96, 202, 54, 295, 137, 17, 79, 214, 413, 454, 305, 187, 4, 458, 330, 290, 73, 220, 118, 125, 180, 247, 243, 257, 194, 117, 320, 104, 252, 87, 95, 228, 324, 271, 398, 334, 148, 425, 190, 78, 151, 34, 310, 122, 376, 102, 260]
	idx = idx[:25]
	# Wm = (255*PlotImage(W_m))
	Wm = W_m - W_m.min()
	alph_est = estimate_normalized_alpha(J, Wm, num_images=15, threshold=125, invert=False, adaptive=False)
	alph = np.stack([alph_est, alph_est, alph_est], axis=2)
	C, est_Ik = estimate_blend_factor(J, Wm, alph)

	alpha = alph.copy()
	for i in range(3):
		alpha[:,:,i] = C[i]*alpha[:,:,i]

	W = Wm.copy()
	for i in range(3):
		W[:,:,i]/=C[i]
	Jt = J[idx]
	Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)

	