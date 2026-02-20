# %%
# Generate some field map
import matplotlib.pyplot as plt
import mr2
import torch

matrix = mr2.data.SpatialDimension(z=1, y=64, x=64)

phantom = mr2.phantoms.EllipsePhantom()
img = phantom.image_space(matrix)


b0_max = 800
b0_map = mr2.phantoms.random_b0map(matrix, fov=matrix * 1e-3, l_max=3, sigma_ppm=1000, seed=1)

plt.imshow(b0_map.squeeze())
plt.colorbar()
plt.show()
# %%
ro_bandwidth = 20e3
t_ro = torch.arange(matrix.x) / ro_bandwidth

fourier_op = mr2.operators.FastFourierOp(dim=(-1, -2))
b0_fourier_op = mr2.operators.ConjugatePhaseFourierOp(fourier_op=fourier_op, b0_map=b0_map, readout_times=t_ro)

(distorted_k,) = b0_fourier_op(img)
(distorted_img,) = fourier_op.H(distorted_k)
vmin, vmax = img.abs().min(), img.abs().max()
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img.abs().squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
ax[0].set_title('Undistorted')
ax[1].imshow(distorted_img.abs().squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
ax[1].set_title('Distorted')
ax[0].axis('off')
ax[1].axis('off')
plt.tight_layout()
# %%


ts_fourier_op = mr2.operators.TimeSegmentedFourierOp(fourier_op=fourier_op, b0_map=b0_map, readout_times=t_ro)

(b0_informed_img,) = ts_fourier_op.H(distorted_k)
(corrected_img,) = mr2.algorithms.optimizers.cg(ts_fourier_op.gram, b0_informed_img)


fig, ax = plt.subplots(1, 2)

ax[0].imshow(b0_informed_img.abs().squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
ax[0].set_title('Time Segmented Adjoint')
ax[1].imshow(corrected_img.abs().squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
ax[1].set_title('Corrected (CG)')
ax[0].axis('off')
ax[1].axis('off')
plt.tight_layout()


# %%
plt.matshow(b0_map.squeeze())
plt.colorbar()
# %%
