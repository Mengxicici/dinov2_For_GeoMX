import os

from dinov2.data.datasets import HEDataset
from dinov2.data.datasets import TiffDataset
import numpy as np
import argparse

def main():
# parse args
    parser = argparse.ArgumentParser(description='dinov2:metadata')
    parser.add_argument('--root', type=str, help='data path of label.txt')
    parser.add_argument('--seg', type=str, help='data path of segementation results')
    parser.add_argument('--names', type=str, help='data path of channelNames')
    parser.add_argument('--picks', type=str, help='data path of channelPicks')
    parser.add_argument('--size', type=int, help='size of window')
    # parser.add_argument('--adds', type=str, help='data apth of HE images')
    parser.add_argument('--out', type=str, help='data path of extras')
    args = parser.parse_args()

# generate dummy labels
    to_file = os.path.join(args.root, "labels.txt")
    with open(to_file, 'w') as f:
        f.write("sample_0,single cell 0\nsample_1,single cell 1\nsample_2,single cell 2\nsample_3,single cell 3\nsample_4,single cell 4\nsample_5,single cell 5\nsample_6,single cell 6\nsample_7,single cell 7\nsample_8,single cell 8\nsample_9,single cell 9\nsample_10,single cell 10")
        #f.write("sample_0,single cell 0\nsample_1,single cell 1")

    for split in TiffDataset.Split:
        dataset = TiffDataset(split=split, root=args.root, extra=args.out, \
                                           seg=args.seg, \
                                           names=args.names, \
                                           picks=args.picks, \
                                           size=args.size
                                           # adds=args.adds
				)
        dataset.dump_extra()

    out0 = np.load(os.path.join(args.out, "entries-TRAIN.npy"))
    print(out0)
    out1 = np.load(os.path.join(args.out, "class-names-TRAIN.npy"))
    print(out1)
    out2 = np.load(os.path.join(args.out, "class-ids-TRAIN.npy"))
    print(out2)

if __name__ == '__main__':
    main()
