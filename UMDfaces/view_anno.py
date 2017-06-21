'''
View annotation for umdfaces VOC_format.
'''
from __future__ import print_function
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

IMAGE_DIR = '/local/mnt/workspace/chris/Databases/UMDFaces/umdfaces_batch2'
ANNO_DIR ='/local/mnt/workspace/chris/Databases/UMDFaces/VOC_format/Annotations_w_bg'

if __name__ == '__main__':

    for anno in os.listdir(ANNO_DIR):
        print('{} '.format(anno),end='')
        fig,ax = plt.subplots(1)
        if anno.endswith('.xml'):
            img_path = os.path.join(IMAGE_DIR,anno.rsplit('_',1)[0],anno.split('.')[0]+'.jpg')
            assert os.path.isfile(img_path)
            tree = ET.parse(os.path.join(ANNO_DIR,anno))
            objs = tree.findall('object')

            img = np.array(Image.open(img_path), dtype=np.uint8)

            # Create figure and axes

            ax.imshow(img)
            for obj in objs:
                bbox = obj.find('bndbox')
                name = obj.find('name').text
                x = float(bbox.find('xmin').text) - 1
                y = float(bbox.find('ymin').text) - 1
                w = float(bbox.find('xmax').text) - 1 - x
                h = float(bbox.find('ymax').text) - 1 - y


                if name == 'face':
                    color = 'r'
                    print('face at ({},{}), ({},{}). '.format(x,y,x+w,y+h),end='')
                elif name == 'background':
                    color = 'g'
                    print('background at ({},{}), ({},{}). '.format(x,y,x+w,y+h))
                else:
                    raise ValueError
                rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor=color,facecolor='none')
                ax.add_patch(rect)
            plt.show(block=False)
            raw_input('Press enter to continue...')
            plt.close()
