
#importing packages

import numpy as np
import cv2


#calculating head pose angle
def get_head_pose(coords):
    #3D face points
	point_3D_17 = np.float32([6.825897, 6.760612, 4.402142])
	point_3D_21 = np.float32([1.330353, 7.122144, 6.903745])
	point_3D_22 = np.float32([-1.330353, 7.122144, 6.903745])
	point_3D_26 = np.float32([-6.825897, 6.760612, 4.402142])
	point_3D_36 = np.float32([5.311432, 5.485328, 3.987654]) 
	point_3D_39 = np.float32([1.789930, 5.393625, 4.413414]) 
	point_3D_42 = np.float32([-1.789930, 5.393625, 4.413414])
	point_3D_45 = np.float32([-5.311432, 5.485328, 3.987654])
	point_3D_31 = np.float32([2.005628, 1.409845, 6.165652]) 
	point_3D_35 = np.float32([-2.005628, 1.409845, 6.165652])
	point_3D_48 = np.float32([2.774015, -2.080775, 5.048531])
	point_3D_54 = np.float32([-2.774015, -2.080775, 5.048531])
	point_3D_57 = np.float32([0.000000, -3.116408, 6.097667]) 
	point_3D_8 = np.float32([0.000000, -7.415691, 4.070434]) 

	object_points = np.float32([point_3D_17,
								point_3D_21,
                               	point_3D_22,
                               	point_3D_26,
                               	point_3D_36,
                               	point_3D_39,
                               	point_3D_42,
                               	point_3D_45,
                               	point_3D_31,
                               	point_3D_35,
                               	point_3D_48,
                               	point_3D_54,
                               	point_3D_57,
                               	point_3D_8])

    #3D projection points source
	reprojection_source = np.float32([[10.0, 10.0, 10.0],
                       				  [10.0, 10.0, -10.0],
                       				  [10.0, -10.0, -10.0],
                       				  [10.0, -10.0, 10.0],
                       				  [-10.0, 10.0, 10.0],
                       				  [-10.0, 10.0, -10.0],
                       				  [-10.0, -10.0, -10.0],
                       				  [-10.0, -10.0, 10.0]])

    #2D face points
	image_points = np.float32([coords[17],
							   coords[21], 
							   coords[22], 
							   coords[26], 
							   coords[36],
							   coords[39], 
							   coords[42], 
							   coords[45], 
							   coords[31], 
							   coords[35],
							   coords[48], 
							   coords[54], 
							   coords[57], 
							   coords[8]])

	K =  [742.11486816, 0.0,            331.6961939,
		  0.0,          726.3464355,    212.95805438,
		  0.0,          0.0,            1.0         ]

    #camera specific camera matrix 
	camera_matrix = np.array(K).reshape(3, 3).astype(np.float32)

    #camera specific distortion coefficients
	distortion_coeffs = np.zeros((5,1), np.float32)

    #solving PnP problem
	_, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coeffs)

    #calculating projection points on 2D image from 3D points source
	reprojected_points, _ = cv2.projectPoints(reprojection_source, rotation_vector, translation_vector, camera_matrix, distortion_coeffs)
	reprojected_points = reprojected_points.reshape(8, 2)

	#calculating eular angle
	rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
	pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
	_, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_matrix)
	return reprojected_points, euler_angle


#adjust size of the frame maintaining its height width ratio
def resize(img, height=None, width=None):
	h, w, _ = img.shape
	if height is None and width is None:
		return img
	if height is None:
		t = w / float(width)
		dim = (width, int(h / t))
	else:
		t = h / float(height)
		dim = (int(w / t), height)
	new_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
	return new_img
