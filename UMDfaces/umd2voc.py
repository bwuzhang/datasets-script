#!/usr/bin/env python

from lxml import etree
import os
import numpy as np
from PIL import Image
import random

FILEDIR = "/local/mnt/workspace/chris/Databases/UMDFaces/umdfaces_batch2/"
IMGSTORE = "/local/mnt/workspace/chris/Databases/UMDFaces/VOC_format/JPEGImages/"
FILENAME = "umdfaces_batch2_ultraface.csv"
ANNOTATIONDIR = "/local/mnt/workspace/chris/Databases/UMDFaces/VOC_format/Annotations_landmarks_9/"

GENERATE_BACKGROUND = False
BACKGROUND_PERCENT = 1.0  #background/faces ratio.

def loadCSVFile(file_name):
    file_content = np.loadtxt(file_name, dtype=np.str, delimiter=",")
    return file_content

def createXML(trans,store_path):
    annotation = etree.Element("annotation")

    folder = etree.SubElement(annotation, "folder" )
    folder.text = trans['folder']

    filename = etree.SubElement(annotation, "filename")
    filename.text = trans['filename']

    source = etree.SubElement(annotation, "source")
    source.text = "Unknown"

    owner = etree.SubElement(annotation, "owner")
    flickrid = etree.SubElement(owner, "flickrid")
    flickrid.text = "NULL"
    name = etree.SubElement(owner, "name")
    name.text = "Unknown"

    size = etree.SubElement(annotation, "size")
    width = etree.SubElement(size, "width")
    width.text = trans['width']
    height = etree.SubElement(size, "height")
    height.text = trans['height']
    depth = etree.SubElement(size, "depth")
    depth.text = trans['depth']

    segmented = etree.SubElement(annotation, "segmented")
    segmented.text = "0"

    _object = etree.SubElement(annotation, "object")
    name = etree.SubElement(_object, "name")
    name.text = "face"
    pose = etree.SubElement(_object, "pose")
    pose.text = "Unspecified"
    truncated = etree.SubElement(_object, "truncated")
    truncated.text = "0"
    difficult = etree.SubElement(_object, "difficult")
    difficult.text = "0"

    bndbox = etree.SubElement(_object, "bndbox")
    xmin = etree.SubElement(bndbox, "xmin")
    xmin.text = trans['xmin']
    ymin = etree.SubElement(bndbox, "ymin")
    ymin.text = trans['ymin']
    xmax = etree.SubElement(bndbox, "xmax")
    xmax.text = trans['xmax']
    ymax = etree.SubElement(bndbox, "ymax")
    ymax.text = trans['ymax']

    landmarks = etree.SubElement(_object,'landmarks')
    d = {}
    # for i in range(21):
    for i in range(9):
        d["l{}".format(i)] = etree.SubElement(landmarks,"l{}".format(i))

        d["l{}x".format(i)] = etree.SubElement(d["l{}".format(i)],"x".format(i))
        d["l{}x".format(i)].text = trans['l'+str(i)+'x']
        d["l{}y".format(i)] = etree.SubElement(d["l{}".format(i)],"y".format(i))
        d["l{}y".format(i)].text = trans['l'+str(i)+'y']
        d["l{}v".format(i)] = etree.SubElement(d["l{}".format(i)],"vis".format(i))
        d["l{}v".format(i)].text = trans['l'+str(i)+'v']

    if GENERATE_BACKGROUND:
        _object_b = etree.SubElement(annotation, "object")
        name_b = etree.SubElement(_object_b, "name")
        name_b.text = "background"
        pose_b = etree.SubElement(_object_b, "pose")
        pose_b.text = "Unspecified"
        truncated_b = etree.SubElement(_object_b, "truncated")
        truncated_b.text = "0"
        difficult_b = etree.SubElement(_object_b, "difficult")
        difficult_b.text = "0"
        bndbox_b = etree.SubElement(_object_b, "bndbox")
        xmin_b = etree.SubElement(bndbox_b, "xmin")
        xmin_b.text = trans['xmin_b']
        ymin_b = etree.SubElement(bndbox_b, "ymin")
        ymin_b.text = trans['ymin_b']
        xmax_b = etree.SubElement(bndbox_b, "xmax")
        xmax_b.text = trans['xmax_b']
        ymax_b = etree.SubElement(bndbox_b, "ymax")
        ymax_b.text = trans['ymax_b']

    tree = etree.ElementTree(annotation)
    file_name = trans['filename'].split('.')[0]+'.xml'
    tree.write(store_path+file_name, pretty_print=True, xml_declaration=True, encoding='UTF-8')

if __name__ == "__main__":
    random.seed(0)
    csv_content = loadCSVFile(FILEDIR+FILENAME)
    #cvs_content_part = csv_content[1:,1:10]
    cvs_content_part = csv_content[1:,1:74]
    i=1
    if GENERATE_BACKGROUND:
        print "Annotation includes random background..."

    for info in cvs_content_part:
        jpg_path = info[0]
        if not os.path.isfile(FILEDIR+jpg_path):
            print (FILEDIR+jpg_path)
        img = Image.open(FILEDIR+jpg_path)
        width, height = img.size

        xmin = float(info[3])
        ymin = float(info[4])
        xmax = float(info[3])+float(info[5])
        ymax = float(info[4])+float(info[6])

        transf = dict()
        transf['folder'] = jpg_path.split('/')[0]
        transf['filename'] = jpg_path.split('/')[1]
        transf['width'] = str(width)
        transf['height'] = str(height)
        transf['depth'] = str(3)
        transf['xmin'] = str(xmin)
        transf['ymin'] = str(ymin)
        transf['xmax'] = str(xmax)
        transf['ymax'] = str(ymax)

        #Read landmarks
        # for i in range(21):
        for i, j in enumerate([7,10,12,14,16,17,18,19,20]):
            transf['l'+str(i)+'x'] = str(float(info[j*3+10]))
            transf['l'+str(i)+'y'] = str(float(info[j*3+11]))
            transf['l'+str(i)+'v'] = str(float(info[j*3+12]))

        while GENERATE_BACKGROUND:
            x1 = random.randint(0,width-1)
            x2 = random.randint(0,width-1)
            y1 = random.randint(0,height-1)
            y2 = random.randint(0,height-1)

            if x1 > x2:
                tmp = x1
                x1 = x2
                x2 = tmp
            if y1 > y2:
                tmp = y1
                y1 = y2
                y2 = tmp
            assert x1 <= x2
            assert y1 <= y2
            x_overlap = max(0,min(x2,xmax)-max(x1,xmin))
            y_overlap = max(0,min(y2,ymax)-max(y1,ymin))
            overlapArea = float(x_overlap) * float(y_overlap)
            if x1==x2 or y1==y2:
                continue
            if (overlapArea/((xmax-xmin)*(ymax-ymin)) < 0.1) and (overlapArea/((x2-x1)*(y2-y1)) < 0.1):
                if random.random() < BACKGROUND_PERCENT:
                    transf['xmin_b'] = str(x1)
                    transf['ymin_b'] = str(y1)
                    transf['xmax_b'] = str(x2)
                    transf['ymax_b'] = str(y2)
                break

        if(i%10000 == 0):
            print "Create No." + str(i) + " XML...."
        createXML(transf,ANNOTATIONDIR)
        i = i + 1


    print "Done..."
