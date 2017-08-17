import random
import os


ANNO_DIR = '../../UMDFaces/VOC_format/Annotations_w_bg'
TRAINING_RATIO = 0.8

if __name__ == '__main__':
    random.seed(0)
    f = []
    for (dirapth,dirnames,filenames) in os.walk(ANNO_DIR):
        f.extend(filenames)
        break
    num_examples = len(f)
    random.shuffle(f)
    num_train = int(TRAINING_RATIO * num_examples)
    train_examples = f[:num_train]
    test_examples = f[num_train:]

    train = open('../../UMDFaces/VOC_format/ImageSets/train.txt','w')
    test = open('../../UMDFaces/VOC_format/ImageSets/test.txt','w')

    for i in train_examples:
        train.write('{}\n'.format(i.split('.')[0]))
    for i in test_examples:
        test.write('{}\n'.format(i.split('.')[0]))
    train.close()
    test.close()
