import os
import sys

import numpy as np

from tifffile import imread, imwrite, TiffFile
from myutils import gen_random_cropped_tiff_to_uint8

def main():
    raw_file = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/Huang/raw/Slide1.ome.tiff"
    img_info_file = "ImageDescription.txt"

    print("Reading " + raw_file)
    with TiffFile(raw_file) as tif:
        t_shape, t_dtype, t_axes = tif.series[0].shape, tif.series[0].dtype, tif.series[0].axes
        imageDescription = tif.pages[0].tags['ImageDescription'].value
    print("The shape, dtype and axes of the raw images are:")
    print(t_shape)
    print(t_dtype)
    print(t_axes)
    print("Saving image description information...")
    with open(img_info_file, 'w') as f:
        f.write(imageDescription)
    print("Image information file is saved to " + img_info_file + "...")
# generate tiff files
    #gen_random_cropped_tiff_to_uint8(raw_file, (0, 1, 2, 3), 4, 1, 0, 20000, 20000, "auto-track", "", "")
if __name__ == '__main__':
    main()
