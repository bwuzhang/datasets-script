'''
Select images with certain parameters
'''
from __future__ import print_function
import os
import numpy as np

VOC_FORMAT_DIR ='/local/mnt2/workspace2/chris/Databases/UMDFaces/VOC_format/'
FILEDIR = "/local/mnt2/workspace2/chris/Databases/UMDFaces/umdfaces_batch2/"
FILENAME = "umdfaces_batch2_ultraface.csv"

def loadCSVFile(file_name):
    file_content = np.loadtxt(file_name, dtype=np.str, delimiter=",")
    return file_content

if __name__ == '__main__':
    csv_content = loadCSVFile(FILEDIR+FILENAME)
    cvs_content_part = csv_content[1:,1:74]
    f = open(VOC_FORMAT_DIR+'batch_2/ImageSets/Right_face.txt', 'w+')
    count = 0
    count_selected = 0
    for info in cvs_content_part:
        count += 1
        jpg_path = info[0]
        if not os.path.isfile(FILEDIR+jpg_path):
            print (FILEDIR+jpg_path)

        yaw = float(info[7])
        if yaw > 30:
            anno_name = info[0].split('.')[0]
            anno_name = anno_name.split('/')[1]
            f.write(anno_name+'\n')
            count_selected += 1

        if count % 1000 == 0:
            print('Scaned {} images, have selected {} images...'.format(count, count_selected))

    print('Scaned {} images, selected {} images.'.format(count, count_selected))
