#image size
#width = 1280, height = 720
# 画像処理用にライブラリをimport 
import pyzed.sl as sl
import numpy as np
import cv2

# PoseEstimator用にライブラリをインポート
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# ソケット通信用にライブラリをインポート
import socket
import json
from time import sleep


sleep_size = 0.01

bbport = 11000
bbhost = '127.0.0.1'
myname = 'name;controller'
targetname = 'sota;'



def main() :
	# create socket object
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try :
		sock.connect((bbhost, bbport))

	except ConnectionRefusedError :
		exit()
	sock.send(myname.encode('utf-8'))
	sleep(sleep_size)

	# Sota setting
	sendmsg = targetname + "setPostureSpeed 300"
	sock.send(sendmsg.encode('utf-8'))
	

	Trans_Matrix = [ 
                   [-0.1, 0, 0], 
                   [0, 0, 0.1],
                   [0, 0.1, 0]
                 ]

	w_pose, h_pose = 1280, 720
	resize_out_ratio = 4.0
	ROI = [1280, 720]
	# create array of Ids
	FACE_POINT_IDS = [0, 14, 15, 16, 17]

	# set OpenPose
	e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w_pose, h_pose))

	#create camera
	zed = sl.Camera()

	#set InitParameters
	init_params = sl.InitParameters()
	init_params.camera_resolution = sl.RESOLUTION.HD720
	init_params.camera_fps = 30
	init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
	init_params.coordinate_units = sl.UNIT.MILLIMETER
	init_params.camera_fps = 60

	# open the camera 
	err = zed.open(init_params)
	if err != sl.ERROR_CODE.SUCCESS :
		print("camera can't open\n")
		exit(1)

	runtime = sl.RuntimeParameters()
	# runtime.enable_depth = False
	image = sl.Mat()
	depth = sl.Mat()
	point_cloud = sl.Mat()

	

	
	



	



	while True :
		if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
			zed.retrieve_image(image, sl.VIEW.LEFT)
			zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
			zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

			# imageデータと画像処理
			image_array = image.get_data()
			image_r = image_array[:,:,0]
			image_g = image_array[:,:,1]
			image_b = image_array[:,:,2]

			image_rgb = np.stack([image_r, image_g, image_b], 2)
			


			h_cam = image.get_height()
			w_cam = image.get_width()
			

			
			# get human pose
			humans = e.inference(image_rgb, upsample_size = resize_out_ratio)

			# extract particular positions
			for hi, hm in enumerate(humans):
				faces_parts = [(max(min(int(bp.x*ROI[0]), ROI[0]), 0), max(min(int(bp.y*ROI[1]), ROI[1]), 0)) 
								for id_, bp in hm.body_parts.items() if id_ in FACE_POINT_IDS]

				if len(faces_parts) == 0 :
					continue

				face_posi = np.average(faces_parts, axis = 0).astype(np.int32)

				# スクリーン座標系
				x = int(face_posi[0])
				y = int(face_posi[1])
				# 白い円を描画
				cv2.circle(image_rgb, (int(face_posi[0]), int(face_posi[1])), 8, (255,255,255), -1)

				# 白い円の位置の三次元座標を取得する
				err, point_cloud_value = point_cloud.get_value(x, y)

				#x,y,z成分を取得
				x_3d = point_cloud_value[0]
				y_3d = point_cloud_value[1]
				z_3d = point_cloud_value[2]
				# 3次元座標位置を出力
				# print("x : {0}, y : {1}, z : {2}".format(x_3d, y_3d, z_3d))

				# Sotaの座標系に変換
				Trans_Matrix = np.array(Trans_Matrix)
				posi_xyz = np.array([x_3d, y_3d, z_3d])
				posi_sota = np.dot(Trans_Matrix, posi_xyz)
				try :
					x_s = str(int(posi_sota[0]))
					y_s = str(int(posi_sota[1]))
					z_s = str(int(posi_sota[2]))
				except ValueError :
					continue
				# print("x : {0}, y : {1}, z : {2}".format(x_s, y_s, z_s))

				# sotaに座標を送る
				sendmsg = targetname + "posture faceTo {0} {1} {2}".format(x_s, y_s, z_s)
				sock.send(sendmsg.encode("utf-8"))
				



            # draw human poses
			image_rgb = TfPoseEstimator.draw_humans(image_rgb, humans, imgcopy=False)
			cv2.imshow("tf_Pose_Estimation_result", image_rgb)
			if cv2.waitKey(5) == ord("q") :
				print("width: {0}, height: {1}".format(image.get_width(), image.get_height()))
				break
		else :
			print("faile grab image\n")
			break

	cv2.destroyAllWindows()
	zed.close()

if __name__ == "__main__" :
	main()
