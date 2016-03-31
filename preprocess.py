from PIL import Image
import argparse
import glob
import os

from dataset import *

CROP_SIZE = 128
# strange scale in the depth data
DEPTH_SCALE_FACTOR = 1e40

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path')
parser.add_argument('processed_dataset_path')

args = parser.parse_args()

# set up folders
try:
    os.mkdir(args.processed_dataset_path)
except:
    pass
try:
    os.mkdir('%s/pos' % args.processed_dataset_path)
    os.mkdir('%s/neg' % args.processed_dataset_path)
except:
    pass

obj_cat = {}
with open('%s/processedData/z.txt' % args.dataset_path) as description_f:
    for line in description_f:
        sid, obj_id, category = line.split()[:3]
        obj_cat[sid] = (obj_id, category)

# file format string
# <pos|neg>/<original image id>-<bounding box id>.<png|tiff>
filename_format = '%s-%03i'

dimg = Image.new('F', (RAW_WIDTH, RAW_HEIGHT))
n_img = 0
n_pos = 0
n_neg = 0
objs = set()
cats = set()
for path in glob.glob('%s/*/pcd*[0-9].txt' % args.dataset_path):
    print path
    n_img += 1
    sample_id = path[-len('1234.txt'):-len('.txt')]
    objs.add(obj_cat[sample_id][0])
    cats.add(obj_cat[sample_id][1])
    dim = convert_pcd(path)
    # replace NaN with 0
    dimg.putdata(np.nan_to_num(dim.flatten() * DEPTH_SCALE_FACTOR))
    with Image.open(path[:-len('.txt')]+'r.png') as cimg:
        # positive grasps
        for i, box in enumerate(read_label_file(path[:-len('.txt')]+'cpos.txt')):
            n_pos += 1
            filename = filename_format % (sample_id, i)
            crop_image(cimg, box, CROP_SIZE).save('%s/pos/%s.png' % (args.processed_dataset_path, filename))
            np.save('%s/pos/%s.npy' % (args.processed_dataset_path, filename), np.reshape(crop_image(dimg, box, CROP_SIZE).getdata(), (CROP_SIZE, CROP_SIZE)))

        # negative grasps
        for i, box in enumerate(read_label_file(path[:-len('.txt')]+'cneg.txt')):
            n_neg += 1
            filename = filename_format % (sample_id, i)
            crop_image(cimg, box, CROP_SIZE).save('%s/neg/%s.png' % (args.processed_dataset_path, filename))
            np.save('%s/neg/%s.npy' % (args.processed_dataset_path, filename), np.reshape(crop_image(dimg, box, CROP_SIZE).getdata(), (CROP_SIZE, CROP_SIZE)))

n_grasp = n_pos + n_neg
print
print 'dataset statistics:'
print '# of objects:', len(objs)
print '# of object categories:', len(cats)
print '# of images:', n_img
print '# of labeled grasps: %i positive: %i (%.2f) negative: %i (%.2f)' % (n_grasp, n_pos, n_pos * 1./n_grasp, n_neg, n_neg * 1./n_grasp)
