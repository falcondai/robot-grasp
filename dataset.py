import numpy as np
from pylab import *
from PIL import Image, ImageChops

RAW_WIDTH = 640
RAW_HEIGHT = 480

# some dataset statistics for the Cornell grasp dataset:
# http://pr.cs.cornell.edu/grasping/rect_data/data.php
# dataset statistics:
# of images: 885
# of labeled grasps: 8019 positive: 5110 (0.64) negative: 2909 (0.36)

def read_label_file(path):
    with open(path) as f:
        xys = []
        has_nan = False
        for l in f:
            x, y = map(float, l.split())
            # XXX some bounding boxes has nan coordinates
            if np.isnan(x) or np.isnan(y):
                has_nan = True
            xys.append((x, y))
            if len(xys) % 4 == 0 and len(xys) / 4 >= 1:
                if not has_nan:
                    yield xys[-4], xys[-3], xys[-2], xys[-1]
                has_nan = False

def convert_pcd(path):
    with open(path) as f:
        # move pass the header
        # http://pointclouds.org/documentation/tutorials/pcd_file_format.php
        for _ in xrange(11):
            f.readline()
            pass
        im = np.zeros((RAW_HEIGHT, RAW_WIDTH), dtype='f4')
        for l in f:
            d, i = l.split()[-2:]
            d = float(d)
            i = int(i)
            x = i % RAW_WIDTH
            y = i / RAW_WIDTH
            im[y, x] = max(0., d)
        return im

def crop_image(img, box, crop_size):
    cx, cy = np.mean(box, axis=0)
    (x1, y1), (x2, y2) = box[:2]
    # center the image to the bounding box
    o = ImageChops.offset(img, int(RAW_WIDTH/2-cx), int(RAW_HEIGHT/2-cy))
    # rotate the gripper axis to the x-axis
    r = o.rotate(np.rad2deg(np.arctan2(y2 - y1, x2 - x1)))
    # crop the image to a fixed size around the bounding box
    return r.crop((RAW_WIDTH/2-crop_size/2, RAW_HEIGHT/2-crop_size/2,
    RAW_WIDTH/2+crop_size/2, RAW_HEIGHT/2+crop_size/2))
