import colorsys
import cv2 as cv
import math
import cunumeric as np
import numpy
from sys import argv
import time

####

# usage: ./vet2.py size(px) zoom_factor(0.x) frames max_iterations(for all pixels) px py (center when to zoom in) julia_cx julia_cy (if no julia_cx/y are given, assuming the mandelbrot is requested)

# ex: ./vet2.py 400 0.95 20 50 0 0 -0.7746806106269039 -0.1374168856037867

###

# all defaults (if no args given)

####
px, py = -0.7746806106269039, -0.1374168856037867 #the poin to zoom in (default one)
px,py = 0,0
max_iteration = 200 
width, height = 1200, 1200
zoom_factor = 0.74 #the inverse of the zoom factor
num_frames = 120 #number of images to create
mandelbrot = 0
julia_cx = -0.835
julia_cx = -1.7693831
julia_cy = -0.2321
julia_cy = 0.00423684
R = 4 #the lenght in the complex plane of width and height, 3 is for mandelbrot, 4 for julia
###

# passed from json/args

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

escape_radius = 0.5 + np.sqrt(1+4*np.sqrt(julia_cx**2 + julia_cy**2))*0.5
###

# this dont change between mandelbrot or a generic julia set

###

mfactor = 0.5
RZF = 1/1000000000000 #min R after that the creation of images will be stopped 

array_frames = []

'''
not really useful, the generic one is almost the same

def color_svg_to_rgb_matrix_s_eq1(matrix_h,matrix_v):
	i = int(matrix_h*6.0) #assuming int() truncates
	return i
'''
# colorsys.hsv_to_rgb() rewritten using numpy vector functions
def color_hsv_to_rgb(matrix_h,matrix_s,matrix_v):

	i = np.where(matrix_s==0,matrix_v,-1)
	f = i
	p = i
	q = i
	t = i
	#start calc i
	i = np.where(matrix_s!=0,6*matrix_h.astype(int),i) #assumes int() truncates
	#start calc f
	f = np.where(f==-1,(matrix_h*6.0)-i,f)
        #start calc p
	pt = 255*matrix_v*(1.0-matrix_s)
	pt = pt.astype(int)
	pt = np.where(pt<255,pt,255)
	p = np.where(p==-1,pt,p)
	p = p.astype(int)
	#start calc q
	qt = 255*matrix_v*(1.0-matrix_s*f)
	qt = qt.astype(int)
	qt = np.where(qt<255,qt,255)
	q = np.where(q==-1,qt,q)
	q = q.astype(int)
	#start calc t
	tt = 255*matrix_v*(1.0-matrix_s*(1.0-f))
	tt = tt.astype(int)
	tt = np.where(tt<255,tt,255)
	t = np.where(t==-1,tt,t)
	t = t.astype(int)
	#convert matrix_v
	matrix_v = 255*matrix_v
	matrix_v = matrix_v.astype(int)
	matrix_v = np.where(matrix_v<255,matrix_v,255)
	r_g_b = np.zeros(len(matrix_s)*3)
	r_g_b = np.reshape(r_g_b, (1,len(matrix_s),3) )
	iii = np.dstack((i,i,i))
	#start operating for i == 0
	r_t = np.left_shift(matrix_v,2)
	g_t = np.left_shift(t,1)
	b_t = np.left_shift(p,4)
	rgb_t = np.dstack((r_t,g_t,b_t))
	

	r_g_b = np.where(iii==[0,0,0], rgb_t, r_g_b )
	#start operating for i == 1
	r_t = np.left_shift(q,2)
	g_t = np.left_shift(matrix_v,1)
	b_t = np.left_shift(t,4)
	rgb_t = np.dstack((r_t,g_t,b_t))
	r_g_b = np.where(iii==[1,1,1], rgb_t, r_g_b)
	#start operating for i == 2
	r_t = np.left_shift(p,2)
	g_t = np.left_shift(matrix_v,1)
	b_t = np.left_shift(t,4)
	rgb_t = np.dstack((r_t,g_t,b_t))
	
	r_g_b = np.where(iii==[2,2,2], rgb_t, r_g_b)
	#start operating for i == 3
	r_t = np.left_shift(p,2)
	g_t = np.left_shift(q,1)
	b_t = np.left_shift(matrix_v,4)
	rgb_t = np.dstack((r_t,g_t,b_t))
	r_g_b = np.where(iii==[3,3,3], rgb_t, r_g_b)
	#start operating for i == 4
	r_t = np.left_shift(t,2)
	g_t = np.left_shift(p,1)
	b_t = np.left_shift(matrix_v,4)
	rgb_t = np.dstack((r_t,g_t,b_t))
	r_g_b = np.where(iii==[4,4,4], rgb_t, r_g_b)
	#start operating for i == 5
	r_t = np.left_shift(t,2)
	g_t = np.left_shift(p,1)
	b_t = np.left_shift(q,4)
	rgb_t = np.dstack((r_t,g_t,b_t))
	r_g_b = np.where(iii==[5,5,5], rgb_t, r_g_b)
	r_g_b = np.reshape(r_g_b, (width,height,3) )
	return r_g_b


def gen_julia_set_image(width,height,mandelbrot=0,nozoom=0):

	RX1, RX2, RY1, RY2 = px-R/2, px+R/2,py-R/2,py+R/2
    
	if nozoom == 1: 
		RX1, RX2, RY1, RY2 = -R/2, R/2,-R/2,R/2
	'''
	width = 10
	height = 10
	RX1 = -2
	RX2 = 2
	RY1 = -2
	RY2 = 2
	'''

	x = numpy.linspace(RX1,RX2,width) #questa con cunumeric dà warning
	y = numpy.linspace(RY1,RY2,height)

	#points = np.transpose(np.asarray([np.tile(x, len(y)), np.repeat(y,len(x))])) se lo faccio in una sola istruzione mi dà warning cunumeric e usa numpy
	#creating the grid of points 
	to_transpose = np.asarray([np.tile(x,len(y)) , np.repeat(y,len(x))])
	points = np.transpose(to_transpose)

	
	#julia_matrix = np.zeros((height,width))-1
	#creating the matrix containing the number of iterations for each point
	#iter_matrix = np.zeros((height,width))
	#creating the matrix containing the colors for each point
	#color_matrix = np.zeros((height,width,3))

	#creating the matrix containing the julia_stability for each point
	julia_stability = np.zeros(width*height,dtype=int)
	for i in range(max_iteration):

		
		x,y = np.hsplit(points,2)
		x = np.transpose(x)
		y = np.transpose(y)
		

		if mandelbrot == 0:
			global julia_cx
			global julia_cy
		else:
			julia_cx = x
			julia_cy = y

		#julia_matrix = np.where(np.power(points,2).sum(axis=1)<=escape_radius**2,np.power(points,2).sum(axis=1),-1)
		julia_matrix = np.where(np.power(points,2).sum(axis=1)<=escape_radius**2,np.power(points,2).sum(axis=1),-1)
		#update julia_stability by the values of the julia_matrix +1 or +0 if the value is -1 in the julia_matrix
		julia_stability = np.where(julia_matrix==-1,julia_stability,julia_stability+1)

		julia_result_x = np.where(julia_matrix!=-1,x**2 - y**2 + julia_cx,escape_radius)
		julia_result_y = np.where(julia_matrix!=-1,2*x*y + julia_cy,escape_radius)

		points = np.hstack((np.transpose(julia_result_x),np.transpose(julia_result_y))) 

	v = julia_stability**mfactor/max_iteration**mfactor
	hv = 0.67-v
	hv = np.where(hv<0,hv+1,hv)
	s = np.ones(len(v))
	v = 1-(v-0.1)**2/0.9**2
#	r,g,b = colorsys.hsv_to_rgb(hv,1,1-(v-0.1)**2/0.9**2) #devo riscrivere hsv_to_rgb con una funzione che prenda in input un array di valori
	#create image matrix from r,g,b arrays
#	color_matrix = np.reshape(np.dstack((hv,hv,v)), (width,height,3)) questa serviva quando ancora non avevo reimplementato hsv_to_rgb vettoriale
	color_matrix = color_hsv_to_rgb(hv,s,v)
#	print(color_matrix.shape)
	return color_matrix

def generate_video_from_images(arr_imgs, pathOut, fps,width,height):
	size = width, height
	frames = []
	for i in range(len(arr_imgs)):
		frames.append(numpy.array(255*arr_imgs[i].astype(np.uint8)))
#                frames.append(255*arr_imgs[i].astype(np.uint8))
	fourcc = cv.VideoWriter_fourcc(*"mp4v")
	out=cv.VideoWriter(pathOut,fourcc,fps, size)
	for i in range(len(frames)):
		out.write(frames[i])
	out.release()
s_time = time.time()
for i in range(num_frames):

	if R < RZF: break
	if i == 0 and mandelbrot == 0:
#		cv.imwrite("img/01/init.png",gen_julia_set_image(width, height,nozoom=1))
		array_frames.append(gen_julia_set_image(width, height,nozoom=1))
	else:
		array_frames.append(gen_julia_set_image(width, height))
#	for avanti in range(200): 
#		mfactor = 0.5 + (1/1000000000000)**0.1/R**0.1
#		R *= zoom_factor
	
	#print(k,mfactor)
	print("mafactor, i :",mfactor, i)
#	if i in range(10): cv.imwrite("img/02/maxIter="+str(max_iteration)+"iter"+str(i+30)+"julia.png", array_frames[i])
	mfactor = 0.5 + (1/1000000000000)**0.1/R**0.1
	R *= zoom_factor
print("video incoming..")
print("arr_len: ", len(array_frames))
#generating video
generate_video_from_images(array_frames, "./video.mp4", 5, width, height)
print("exec time: ",(time.time()-s_time) )
