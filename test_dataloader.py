import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# os.chdir(sys.path[0])
# print(os.getcwd())
from DenseConvNet.scripts.model.dataloader.datagen import DataLoader

da = DataLoader(
    "/Users/souvik/Neurogazer/projects/dataset/S3_backup",
    # "/Volumes/Seagate_backup/datasets",
    "train",
    plane="sagittal",
    batch_size=8,
)

# x = da.get_test_volume()
# print(x.shape)
# print(x.max(), x.min())
(a, b, c), (d, e) = da.get_generator_().__getitem__(1)
print(a.shape, b.shape, c.shape, d.shape, e.shape)
# print(a.max(), a.min())
# print(np.unique(b), np.unique(c))

A = np.split(a, 8, axis=0)
B = np.split(b, 8, axis=0)
C = np.split(c, 8, axis=0)
D = np.split(d, 8, axis=0)

lists = [a[6], b[6], d[6], e[6], c[6, :, :, 1], c[6, :, :, 2]]
# print(type(A))
# print(c[0].max(), c[0].min())
# # print(d[0].max(), d[0].min())

fig, axs = plt.subplots(2, 3, figsize=(10, 12))
for j, i in enumerate(axs.flatten()):
    i.imshow(lists[j], cmap="gray")
    i.set_xticks([])

    i.set_yticks([])

plt.tight_layout()
plt.show()


# from scipy.ndimage import laplace
# import nibabel as nib
# from skimage.filters import threshold_local
# import cv2


# img1 = nib.load(
#     "/Users/souvik/Desktop/test/subjects/R_AH0313_KRAA_000616_01/brain_extraction/T1w_acpc.nii.gz"
# ).get_fdata()
# # img1 = (img - img.min()) / (img.max() - img.min())

# # img_1 = laplace(img1)
# # img_11 = np.abs(img_1)
# # img_111 = (img_11 - img_11.min()) / (img_11.max() - img_11.min())

# img_1 = cv2.Laplacian(img1, ddepth=-1, ksize=3)
# img_11 = np.abs(img_1)
# img_111 = (img_11 - img_11.min()) / (img_11.max() - img_11.min())

# img_2 = cv2.Laplacian(img1, ddepth=-1, ksize=5)
# img_21 = np.abs(img_2)
# img_211 = (img_21 - img_21.min()) / (img_21.max() - img_21.min())

# img_3 = cv2.Laplacian(img1, ddepth=-1, ksize=7)
# img_31 = np.abs(img_1)
# img_311 = (img_31 - img_31.min()) / (img_31.max() - img_31.min())

# img_th1 = threshold_local(img_111, block_size=3)
# maxes1 = np.amax(img_th1, axis=(1, 2), keepdims=True)
# img_th1 = img_th1 / maxes1

# img_th2 = threshold_local(img_211, block_size=5)
# maxes2 = np.amax(img_th2, axis=(1, 2), keepdims=True)
# img_th2 = img_th2 / maxes2

# img_th3 = threshold_local(img_311, block_size=7)
# maxes3 = np.amax(img_th3, axis=(1, 2), keepdims=True)
# img_th3 = img_th3 / maxes3


# n = 100
# xx = (
#     0.33 * np.abs(img_th1[:, :, n])
#     + 0.33 * np.abs(img_th3[:, :, n])
#     + 0.34 * np.abs(img_th2[:, :, n])
# )

# lists = [
#     np.abs(img1[:, :, n]),
#     np.abs(img_th2[:, :, n]),
#     np.abs(img_th3[:, :, n]),
#     np.abs(img_th1[:, :, n]) + np.abs(img_th2[:, :, n]),
#     np.abs(img_th2[:, :, n]) + np.abs(img_th3[:, :, n]),
#     xx,
# ]


# fig, axs = plt.subplots(2, 3, figsize=(10, 12))
# for j, i in enumerate(axs.flatten()):
#     i.imshow(lists[j], cmap="gray")
#     i.set_xticks([])

#     i.set_yticks([])

# plt.tight_layout()
# plt.show()
