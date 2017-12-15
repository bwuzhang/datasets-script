import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

IMAGE_FOLDER = '/local/mnt2/workspace2/chris/Databases/CelebA/img_celeba/'
ANNO_FOLDER = '/local/mnt2/workspace2/chris/Databases/CelebA/Anno/'


landmark_anno_file = open(ANNO_FOLDER + 'list_landmarks_celeba.txt', 'r')
bbox_anno_file = open(ANNO_FOLDER + 'list_bbox_celeba.txt', 'r')

# Discard first two lines
landmark_anno_file.readline()
landmark_anno_file.readline()
bbox_anno_file.readline()
bbox_anno_file.readline()
# for i in range(10):
while(True):
    landmark_line =  landmark_anno_file.readline()
    image_name = landmark_line.split(' ')[0]
    landmark_numbers = [int(s) for s in landmark_line.split() if s.isdigit()]

    bbox_line = bbox_anno_file.readline()
    assert(landmark_line[0] == bbox_line[0])
    bbox_number = [int(s) for s in bbox_line.split() if s.isdigit()]

    img = np.array(Image.open(IMAGE_FOLDER + image_name), dtype=np.uint8)

    # bbox
    x = float(bbox_number[0])
    y = float(bbox_number[1])
    w = float(bbox_number[2])
    h = float(bbox_number[3])
    print bbox_number

    # landmark
    l_x = []
    l_y = []
    for j in range(5):
        l_x.append(float(landmark_numbers[j*2]))
        l_y.append(float(landmark_numbers[j*2+1]))

    fig,ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.plot(l_x,l_y,'ro')

    plt.show(block=False)
    raw_input('{} Press enter to continue...'.format(image_name))
    plt.close()
