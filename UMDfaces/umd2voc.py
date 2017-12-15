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


landmark_dic = {'7':'0', '14':'1', '10':'2', '17':'3', '18':'4', '19':'5', '12':'6', '20':'7', '16':'8'}

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

    tree = etree.ElementTree(annotation)
    file_name = trans['filename'].split('.')[0]+'.xml'
    tree.write(store_path+file_name, pretty_print=True, xml_declaration=True, encoding='UTF-8')

if __name__ == "__main__":
    random.seed(0)
    csv_content = loadCSVFile(FILEDIR+FILENAME)
    #cvs_content_part = csv_content[1:,1:10]
    cvs_content_part = csv_content[1:,1:74]
    i=1
    
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
            transf['l'+landmark_dic[str(j)]+'x'] = str(float(info[j*3+10]))
            transf['l'+landmark_dic[str(j)]+'y'] = str(float(info[j*3+11]))
            transf['l'+landmark_dic[str(j)]+'v'] = str(float(info[j*3+12]))

        if(i%10000 == 0):
            print "Create No." + str(i) + " XML...."
        createXML(transf,ANNOTATIONDIR)
        i = i + 1


    print "Done..."
