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

# IMAGE_DIR = '/local/mnt/workspace/chris/Databases/datasets-script/UMDfaces/demo_image'
IMAGE_DIR = '/local/mnt2/workspace2/chris/Databases/UMDFaces/umdfaces_batch2/charlie_morgan'
ANNO_DIR ='/local/mnt2/workspace2/chris/Databases/UMDFaces/VOC_format/batch_2/Annotations'

def draw_image_anno(img_path, anno_path):

    img = np.array(Image.open(img_path), dtype=np.uint8)
    tree = ET.parse(anno_path)
    objs = tree.findall('object')
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

        landmarks = obj.find('landmarks')
        l_x = []
        l_y = []
        # for i in [7,10,12,14,16,17,18,19,20]:
        for i in range(9):
            l_x.append(float(landmarks.find('l'+str(i)).find('x').text))
            l_y.append(float(landmarks.find('l'+str(i)).find('y').text))

        ax.plot(l_x,l_y,'ro')
    plt.show(block=False)
    raw_input('Press enter to continue...')
    plt.close()


# if __name__ == '__main__':

#     for anno in os.listdir(ANNO_DIR):
#         print('{} '.format(anno),end='')
#         fig,ax = plt.subplots(1)
#         if anno.endswith('.xml'):
#             img_path = os.path.join(IMAGE_DIR,anno.rsplit('_',1)[0],anno.split('.')[0]+'.jpg')
#             assert os.path.isfile(img_path)
#             img = np.array(Image.open(img_path), dtype=np.uint8)
#             draw_image_anno(img, os.path.join(ANNO_DIR,anno))

# if __name__ == '__main__':
#
#     for img_path in os.listdir(IMAGE_DIR):
#         # print('{} '.format(img_path),end='')
#         fig,ax = plt.subplots(1)
#         if img_path.endswith('.jpg'):
#             anno_path = os.path.join(ANNO_DIR,img_path.split('.')[0]+'.xml')
#             print(anno_path)
#             assert os.path.isfile(anno_path)
#             draw_image_anno(os.path.join(IMAGE_DIR,img_path), anno_path)

if __name__ == '__main__':

    for img_path in os.listdir(IMAGE_DIR):
        # print('{} '.format(img_path),end='')
        fig,ax = plt.subplots(1)
        if img_path.endswith('.jpg'):
            anno_path = os.path.join(ANNO_DIR,img_path.split('.')[0]+'.xml')
            print(anno_path)
            if not os.path.isfile(anno_path):
                continue
            draw_image_anno(os.path.join(IMAGE_DIR,img_path), anno_path)
