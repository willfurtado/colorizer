"""Main file for CS194-26 Project 1."""
import os
import sys

import numpy as np
import skimage.io as skio
from utils import align_full_image, convert_to_bgr_channels

if __name__ == "__main__":
    for imname in os.listdir("data"):
        if not imname.endswith(".jpg") and not imname.endswith(".tif"):
            continue
        b, g, r = convert_to_bgr_channels(filepath=f"data/{imname}")

        num_layers = int(np.ceil(np.log2(b.shape[0] / 200)))

        (
            im_out,
            (g_x_shift, g_y_shift),
            (r_x_shift, r_y_shift),
        ) = align_full_image(b=b, g=g, r=r, depth=num_layers)

        sys.stdout.write(f"{imname} Green Shift: {(g_x_shift, g_y_shift)}")
        sys.stdout.write(f"{imname} Red Shift: {(r_x_shift, r_y_shift)}")

        # save the image
        fname = f"output/{imname.replace('tif', 'jpg')}"
        skio.imsave(fname, im_out)
