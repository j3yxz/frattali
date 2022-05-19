from PIL import Image

import colorsys

import math



px, py = -0.7746806106269039, -0.1374168856037867 #Tante Renate

R = 3 

max_iteration = 2500

w, h = 1024,1024

mfactor = 0.5



def Mandelbrot(x,y,max_iteration,minx,maxx,miny,maxy):

    zx = 0

    zy = 0

    RX1, RX2, RY1, RY2 = px-R/2, px+R/2,py-R/2,py+R/2

    cx = (x-minx)/(maxx-minx)*(RX2-RX1)+RX1

    cy = (y-miny)/(maxy-miny)*(RY2-RY1)+RY1

    i=0

    while zx**2 + zy**2 <= 4 and i < max_iteration:

        temp = zx**2 - zy**2

        zy = 2*zx*zy + cy

        zx = temp + cx

        i += 1

    return i



def gen_Mandelbrot_image(sequence):

  bitmap = Image.new("RGB", (w, h), "white")

  pix = bitmap.load()

  for x in range(w):

    for y in range(h):

      c=Mandelbrot(x,y,max_iteration,0,w-1,0,h-1)

      v = c**mfactor/max_iteration**mfactor

      hv = 0.67-v

      if hv<0: hv+=1

      r,g,b = colorsys.hsv_to_rgb(hv,1,1-(v-0.1)**2/0.9**2)

      r = min(255,round(r*255))

      g = min(255,round(g*255))

      b = min(255,round(b*255))

      pix[x,y] = int(r) + (int(g) << 8) + (int(b) << 16)

  bitmap.save("Mandelbrot_"+str(sequence)+".jpg")



R=3

f = 0.975

RZF = 1/1000000000000

k=1

while R>RZF:

  if k>100: break

  mfactor = 0.5 + (1/1000000000000)**0.1/R**0.1

  print(k,mfactor)

  gen_Mandelbrot_image(k)

  R *= f

  k+=1
