import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import cv2 as cv
#######this is from opencv c++ code




def complex_matrix(xmin, xmax, ymin, ymax, width, height):
	#scaleX = (width/int(xmax-xmin))
	#scaleY = (width/int(ymax-ymin))

    re = np.linspace(xmin, xmax, width)
    im = np.linspace(ymin, ymax, height)
    #return re[np.newaxis, :] + im[:, np.newaxis] * 1j
    return re, im
def is_stable(c, num_iterations):
    z = 0
    for _ in range(num_iterations):
        z = z ** 2 + c
    return abs(z) <= 2

def get_members(c, num_iterations):
    mask = is_stable(c, num_iterations)
    return c[mask]

print("argv: ", argv[1])

c = complex_matrix(-2, 0.5, -1.5, 1.5, 4000, 4000)
#print(c)
#members = get_members(c, num_iterations=int(argv[1]))

#non lo voglio stampare, ma nel caso:
'''
plt.scatter(members.real, members.imag, color="black", marker=",", s=1)
plt.gca().set_aspect("equal")
plt.axis("off")
plt.tight_layout()
plt.show()
'''





from dataclasses import dataclass

@dataclass
class MandelbrotSet:
    max_iterations: int

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def stability(self, c: complex) -> float:
        return self.escape_count(c) / self.max_iterations

    def escape_count(self, c: complex) -> int:
        z = 0
        for iteration in range(self.max_iterations):
            z = z ** 2 + c
            if abs(z) > 2:
                return iteration
        return self.max_iterations
    
    def sqrt_stability(self, c: complex) -> int:
    	value = float(self.escape_count(c))
    	if(int(value)-self.max_iterations == 0):
    		return 0
    	elif(int(value)==0):
    		return 255
    	else:
    		return int( ((1 - value/self.max_iterations)**0.5)*255 +0.5)

    def palette_x_channel(self, c: complex):
    	integer_to_palette = self.sqrt_stability(c)
    	R, G , B = integer_to_palette
    	return R,G,B
def create_set(c,max_iter,width,height):
	#il magic number dovrà diventare il channel dei colori in qualche modo
	#ret = np.empty_like([[0],[1]], dtype='float64')
	ret = np.zeros((3, 3, 3), dtype = "uint8")
	ret.resize(width,height,3)
	#da impostare a max_iter
	mandelbrotSet = MandelbrotSet(max_iterations=int(max_iter))
	re, im = c
	#print(re.shape)
	for i in range(re.shape[0]):
		for j in range(im.shape[0]):
	#		print(re[i], im[j])
			ret[j][i][0] = mandelbrotSet.sqrt_stability(complex(re[i]+im[j]*1j))
			ret[j][i][1] = ret[j][i][0]
			ret[j][i][2] = ret[j][i][0]
	return ret


max_iter = argv[1]
width, height = 4000, 4000
#np.array([width,height,1])




###############################
mandelbrot_set = MandelbrotSet(max_iterations=20)
'''
scale = 0.0075
GRAYSCALE = "L"
members = []
print(c.size, c.shape[0], c.shape[1], c)
print(c[0][0], c[3][3])
a = np.array([width, height, 3])
image = np.empty_like(a)

for i in range(c.real):
    for j in range(c.imag):
        p = mandelbrot_set.stability(c[i][j])
        pc = int(p * 255 / 3)
        np.append(image,pc)
        np.append(image,pc)
        np.append(image,pc)

#for index, pixel in enumerate(c):
#to remove
'''
print("starting to write..")
    
image_row = create_set(c, max_iter, width, height)

#perché?
#cv.imread(image, GRAYSCALE)
def zoom_on_coord(xc, yc, zoom_factor, img):
#calc the images parameters
	width = img.shape[1]
	height = img.shape[0]
#calc new borders
	x1 = int(xc-0.5*width*(1-1/zoom_factor))
	x2 = int(xc+0.5*width*(1-1/zoom_factor))
	y1 = int(yc-0.5*height*(1-1/zoom_factor))
	y2 = int(yc+0.5*height*(1-1/zoom_factor))
	img_cropped = img[y1:y2,x1:x2]
	print(img)
	return 1#cv.resize(img, None, fx=zoom_factor, fy=zoom_factor)

image = image_row # cv.imread(image_row, cv.IMREAD_GRAYSCALE)
#image_z = zoom_on_coord(100, 100, 1.3, image)
cv.imwrite("mandelbrot.png", image)
#cv.imwrite("z.png", image_z)
#for x in c.split(" "):
#    print(x)
    #members.append(mandelbrot_set.stability(x))

#membersGreyScale = mandelbrot_set.stability(c)
#print(members)
########################

'''
from viewport import Viewport
from PIL import Image
image = Image.new(mode=GRAYSCALE, size=(width, height))
for y in range(height):
    for x in range(width):
        c = scale * complex(x - width / 2, height / 2 - y)
        instability = 1 - mandelbrot_set.stability(c)
        image.putpixel((x, y), int(instability * 255))

image.show()
#print(members)
'''
#############################################
# part in which will be generated the video from images,
# probably will be moved to another script
###############################
def gen_multiply_images(cx,cy,zoom_factor,steps,width,height,iter_feach_image):
	x0,x1,y0,y1 = -2, 0.5, -1.5, 1.5
	mandelbrot_set = MandelbrotSet(max_iterations=iter_feach_image)
	arr_images = []
	for i in range(steps):
		c = complex_matrix(x0,x1,y0,y1, width, height)
		image_row = create_set(c, max_iter, width, height)
		arr_images.append(image_row)
		'''
		x0 = int(cx-0.5*width*(1-1/zoom_factor))
		x1 = int(cx+0.5*width*(1-1/zoom_factor))
		y0 = int(cy-0.5*height*(1-1/zoom_factor))
		y1 = int(cy+0.5*height*(1-1/zoom_factor))
		'''
		#x0-cx < x1 - cx ? mini, maxi = x0-cx, x1-cx : maxi, mini = x0-cx, x1-cx bel tentativo..  
		sx = False
		maxi, mini = cx-x0, x1-cx
		if cx - x0 < x1 - cx :
			mini,maxi = maxi,mini
			sx = True
		ratiomm=maxi/mini
		if ratiomm > 1.2:
			to_move = ratiomm/10
			if sx == True:
				x0 -= to_move
				x1 -= to_move
				y0 -= to_move
				y1 -= to_move
			else:
				x0 += to_move
				x1 += to_move
				y0 += to_move
				y1 += to_move
		x0 /= zoom_factor
		x1 /= zoom_factor
		y0 /= zoom_factor
		y1 /= zoom_factor

	return arr_images

def generate_video_from_images(arr_imgs, pathOut, fps, time):
	frames = []
	for i in range(arr_imgs.shape[0]):
		frames.append(arr_imgs[i])
	out=cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4'), (fps, size))
	for i in range(len(frames)):
		out.write(frames[i])
	out.release()

out = gen_multiply_images(-1,0)