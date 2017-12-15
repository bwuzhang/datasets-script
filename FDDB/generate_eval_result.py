import sys
import numpy as np
import os

num_images_in_each_fold = [290, 285, 274, 302, 298, 302, 279, 276, 259, 280]

def convert_to_fddb_format(image_ids, detection_boxes, detection_scores):
	assert len(set(image_ids)) == 2845
	assert len(image_ids) == 2845
	assert len(detection_boxes) == 2845
	assert len(detection_scores) == 2845

	cur_idx = 0
	for i in range(10):
		print('Processing FDDB-fold-{}'.format(i+1, '02'))
		cur_image_ids = image_ids[cur_idx:num_images_in_each_fold[i] + cur_idx]
		cur_detection_boxes = detection_boxes[cur_idx:num_images_in_each_fold[i] + cur_idx]
		cur_detection_scores = detection_scores[cur_idx:num_images_in_each_fold[i] + cur_idx]

		assert (len(cur_detection_scores) == len(cur_image_ids)) and (len(cur_detection_scores) == len(cur_detection_boxes))

		file = open(os.path.join(input_dir, 'fold-'+format(i+1, '02')+'-out.txt'), 'w+')

		for (image_id, detection_boxes_per_image, detection_scores_per_image) in zip(cur_image_ids, cur_detection_boxes, cur_detection_scores):
			assert len(detection_boxes_per_image) == len(detection_scores_per_image)
			file.write(image_id+'\n')
			file.write(str(len(detection_boxes_per_image))+'\n')
			for (detection_box, detection_score) in zip(detection_boxes_per_image, detection_scores_per_image):
				detection_str = str(detection_box[1]) + ' ' + str(detection_box[0]) + \
					' ' + str(detection_box[3]-detection_box[1])+' '+str(detection_box[2]-detection_box[0]) + \
					' ' + str(detection_score) + '\n'
				dummy = 1 + 5
				file.write(detection_str)
		file.close()
		cur_idx += num_images_in_each_fold[i]


if __name__ == '__main__':
	input_dir = sys.argv[1]
	image_ids = np.load(os.path.join(input_dir, 'image_ids.npy'))
	detection_boxes = np.load(os.path.join(input_dir, 'detection_boxes.npy'))
	detection_scores = np.load(os.path.join(input_dir, 'detection_scores.npy'))
	convert_to_fddb_format(image_ids, detection_boxes, detection_scores)
