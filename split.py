import glob
from pylab import *
import os
from os import path

def ln(fns, dst_dir):
    for p in fns:
        fn = path.basename(p)
        d = path.dirname(p)
        name = fn[:-len('.png')]
        os.symlink(p, path.join(dst_dir, fn))
        os.symlink(path.join(d, name + '.npy'), path.join(dst_dir, name + '.npy'))

np.random.seed(0)

# create train/val splits
pos_fns = glob.glob('processed/pos/*.png')
neg_fns = glob.glob('processed/neg/*.png')
pos_fns.sort()
neg_fns.sort()
shuffle(pos_fns)
shuffle(neg_fns)

train_pos = pos_fns[::2]
train_neg = neg_fns[::2]
val_pos = pos_fns[1::2]
val_neg = neg_fns[1::1]

# set up train/val split folders
os.mkdir('splits')

os.makedirs('splits/train/pos')
os.makedirs('splits/train/neg')
os.makedirs('splits/val/pos')
os.makedirs('splits/val/neg')

# make symbolic links
ln(train_pos, 'splits/train/pos')
ln(train_neg, 'splits/train/neg')
ln(val_pos, 'splits/val/pos')
ln(val_neg, 'splits/val/neg')

# create a randomly permutated list
train_fn = train_pos + train_neg
train_y = [1] * len(train_pos) + [0] * len(train_neg)
ii = np.random.permutation(len(train_fn))
with open('splits/train_fn.txt', 'wb') as f:
    for i in ii:
        p = train_fn[i]
        name = p[:-len('.png')]
        f.write('%s %s\n' % (p, name+'.npy'))
np.save('splits/train_y.npy', asarray(train_y)[ii])

val_fn = val_pos + val_neg
val_y = [1] * len(val_pos) + [0] * len(val_neg)
# ii = np.random.permutation(len(val_fn))
with open('splits/val_fn.txt', 'wb') as f:
    for p in val_fn:
        name = p[:-len('.png')]
        f.write('%s %s\n' % (p, name+'.npy'))
np.save('splits/val_y.npy', asarray(val_y))
