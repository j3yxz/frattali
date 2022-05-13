import numpy as np
#import matplotlib.pyplot as plt
from sys import argv

def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
    re = np.linspace(xmin, xmax, int((xmax - xmin) * pixel_density))
    im = np.linspace(ymin, ymax, int((ymax - ymin) * pixel_density))
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j

def is_stable(c, num_iterations):
    z = 0
    for _ in range(num_iterations):
        z = z ** 2 + c
    return abs(z) <= 2

def get_members(c, num_iterations):
    mask = is_stable(c, num_iterations)
    return c[mask]

print("argv: ", argv[1])

c = complex_matrix(-2, 0.5, -1.5, 1.5, pixel_density=300)
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


###############################
mandelbrot_set = MandelbrotSet(max_iterations=20)
width, height = 512, 512
scale = 0.0075
GRAYSCALE = "L"
members = []
print(c.size)

a = np.array([[1, 2, 3],[4,5,6]])
image = np.empty_like(a)

for i in range(c.shape[0]):
    for j in range(c.shape[1]):
        p = mandelbrot_set.stability(c[i][j])
        pc = int(p * 255 / 3)
        np.append(image,pc)
        np.append(image,pc)
        np.append(image,pc)

#for index, pixel in enumerate(c):

    

import cv2 as cv
cv.read_image(image, GRAYSCALE)
cv.imwrite("mandelbrot.jpg", image)
#for x in c.split(" "):
#    print(x)
    #members.append(mandelbrot_set.stability(x))

#membersGreyScale = mandelbrot_set.stability(c)
print(members)
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
