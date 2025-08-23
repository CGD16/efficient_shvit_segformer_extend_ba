# Create a vedo Volume from the NumPy array
# Create a vedo Volume from the NumPy array
import numpy as np
from vedo import dataurl, Plotter, Volume, show

file_path = "/mnt/d/Datasets/shoes/images_3d/images_3d_SegFormer_2conv_320x320x320/volumes/Bruschi_down2_2_2.npy"


# Load your 3D NumPy array
volume_array = np.load(file_path)  # shape: (Z, Y, X)

volume_array = volume_array[200:,:,:]

threshold = 0.2  # adjust this based on your data
volume_array[volume_array < threshold] = 0

# Create a vedo Volume from the NumPy array
vol = Volume(volume_array)
vol.color("coolwarm").alpha([0, 0.1, 0.3, 0.6, 1.0])  # control transparency

# Show the volume with an interactive window
show(vol, axes=1, bg='white')

