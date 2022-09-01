"""Main file for CS194-26 Project 1."""
import os

import numpy as np
import skimage.io as skio
from utils import align_full_image, convert_to_bgr_channels

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("files", nargs="+")
    # args = parser.parse_args()

    print(f"Runnning for: {os.listdir('data')}")
    for imname in os.listdir("data"):
        if not imname.endswith(".jpg") and not imname.endswith(".tif"):
            continue
        b, g, r = convert_to_bgr_channels(filepath=f"data/{imname}")

        num_layers = int(np.ceil(np.log2(b.shape[0] / 200)))

        im_out = align_full_image(b=b, g=g, r=r, depth=num_layers)

        # save the image
        fname = f"output/{imname.replace('tif', 'jpg')}"
        skio.imsave(fname, im_out)
