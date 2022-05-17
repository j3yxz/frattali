import cv2
import os
from o.path import isfile, join


def convert_pictures_to_video(in_images,pathOut, fps, time):
	frame_array=[]
	files = in_images
	for i in range(files):
		img=cv2.imread(files[i])
		#da spostare fuori visto che le genero e hanno le stesse dimensioni
		height, width, layers = img.shape
		size = (width, height)

		for k in range(time):
			frame_array.append(img)
			#forse da sostituire il secondo argomento con -1 (?)
	out=cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4'), fps, size)
	for i in range(len(frame_array)):
		out.write(frame_array[i])
	out.release()
