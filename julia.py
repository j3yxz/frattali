import colorsys
import cv2 as cv
import math
import numpy as np
from sys import argv

####

# all defaults

####
px, py = -0.7746806106269039, -0.1374168856037867 #the poin to zoom in (default one)
max_iteration = 50
width, height = 400, 400
zoom_factor = 0.97 #the inverse of the zoom factor
num_frames = 15 #number of images to create
mandelbrot = 0
julia_cx = -0.835
julia_cy = -0.2321
R = 3 #the lenght in the complex plane of width and height, 3 is for mandelbrot, 4 for julia
###

# passed from json

###
print(argv)
if len(argv) > 4:
	width = int(argv[1])
	height = int(argv[1])
	zoom_factor = float(argv[2])
	num_frames = int(argv[3])
	max_iteration = int(argv[4])
	px  = float(argv[5])
	py = float(argv[6])
if len(argv) > 7:
	julia_cx  = float(argv[7])
	julia_cy = float(argv[8]) 
	escape_radius = 0.5 + np.sqrt(1+4*np.sqrt(julia_cx**2 + julia_cy**2))*0.5
	mandelbrot = 0
	R = 4
else : #this is mandelbrot requested
	mandelbrot = 1
	escape_radius = 2

###

# this dont change between mandelbrot or a generic julia set

###

mfactor = 0.5
RZF = 1/1000000000000 #min R after that the creation of images will be stopped (probably not to use actually)

array_frames = []

def julia_stability(x,y,max_iteration,minx,maxx,miny,maxy,mandelbrot,nozoom=0):
    zx = 0
    zy = 0
    RX1, RX2, RY1, RY2 = px-R/2, px+R/2,py-R/2,py+R/2
    
    if nozoom == 1: 
    	RX1, RX2, RY1, RY2 = -R/2, R/2,-R/2,R/2

    cx = (x-minx)/(maxx-minx)*(RX2-RX1)+RX1
    cy = (y-miny)/(maxy-miny)*(RY2-RY1)+RY1

    if mandelbrot == 0:
    	global julia_cx
    	global julia_cy
    else :
    	julia_cx = cx
    	julia_cy = cy
    zx = cx
    zy = cy
    i=0

    while zx**2 + zy**2 <= escape_radius**2 and i < max_iteration:

        temp = zx**2 - zy**2

        zy = 2*zx*zy + julia_cy

        zx = temp + julia_cx

        i += 1

    return i



def gen_julia_set_image(width,height,nozoom=0):

	image = np.array((width,height,3), dtype=int)
	image.resize(width,height,3)
	for x in range(width):

		for y in range(height):

			c=julia_stability(x,y,max_iteration,0,width-1,0,height-1,mandelbrot,nozoom)

			v = c**mfactor/max_iteration**mfactor

			hv = 0.67-v

			if hv<0: hv+=1

			r,g,b = colorsys.hsv_to_rgb(hv,1,1-(v-0.1)**2/0.9**2)

			r = min(255,round(r*255))

			g = min(255,round(g*255))

			b = min(255,round(b*255))
			#the << x mix colors a bit, need to find the best one
			image[y][x][0], image[y][x][1], image[y][x][2] = int(r), (int(g) << 1), (int(b) << 1)

	return image

def generate_video_from_images(arr_imgs, pathOut, fps,width,height):
	size = width, height
	frames = []
	for i in range(len(arr_imgs)):
		frames.append(255*arr_imgs[i].astype(np.uint8))
	fourcc = cv.VideoWriter_fourcc(*"mp4v")
	out=cv.VideoWriter(pathOut,fourcc,fps, size)
	for i in range(len(frames)):
		out.write(frames[i])
	out.release()


for i in range(num_frames):

	if R < RZF: break
	if i == 0 and mandelbrot == 0:
#		cv.imwrite("img/01/init.png",gen_julia_set_image(width, height,nozoom=1))
		array_frames.append(gen_julia_set_image(width, height,nozoom=1))
	else:
		array_frames.append(gen_julia_set_image(width, height))
#	for avanti in range(5): 
#		R *= zoom_factor
#		mfactor = 0.5 + (1/1000000000000)**0.1/R**0.1
	
	#print(k,mfactor)
	print("mafactor 88:",mfactor)
#	if i in range(10): cv.imwrite("img/02/maxIter="+str(max_iteration)+"iter"+str(i+30)+"julia.png", array_frames[i])
	mfactor = 0.5 + (1/1000000000000)**0.1/R**0.1
	R *= zoom_factor
print("video incoming..")
print("arr_len: ", len(array_frames))
#generating video
generate_video_from_images(array_frames, "./video.mp4", 3, width,height)