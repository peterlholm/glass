#!/usr/bin/env python

'''
Stitching sample
================

Show how to use Stitcher API from python in a simple way to stitch panoramas
or scans.
'''
# pylint: disable=import-error, wrong-import-order,consider-using-f-string,missing-function-docstring

# Python 2/3 compatibility
from __future__ import print_function

from pathlib import Path
#import numpy as np
import cv2 as cv
#import argparse
import sys

__doc__ += '\n'

def stitch_test(folder: Path, result: Path):
    files = list(folder.glob('*.png')) + list(folder.glob('*.PNG')) + list(folder.glob('*.jpg')) + list(folder.glob('*.JPG'))
    # read input images
    imgs = []

    for img_name in files:
        print(img_name)

        img = cv.imread(str(img_name))
        if img is None:
            print("can't read image " + img_name)
            sys.exit(-1)
        imgs.append(img)

    print ("antal images",len(imgs))
    if len(imgs)<=1:
        print('missing files')
        sys.exit(-1)
    mode = cv.Stitcher_PANORAMA
    #mode = cv.Stitcher_SCAN

    stitcher = cv.Stitcher.create(mode)
    status, pano = stitcher.stitch(imgs)
    print (stitcher.workScale)


    print("Status", status)
    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)

    cv.imwrite(str(result), pano)
    print(f"stitching completed successfully. {result} saved!")

    print('Done')


if __name__ == '__main__':
    #print(__doc__)
    #main()
    folder = Path("testdata/test4")
    print(folder)
    result = folder / 'result.jpg'
    stitch_test(folder, result)

    cv.destroyAllWindows()
