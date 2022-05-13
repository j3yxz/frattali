# create a function that computes mandelbrot set for a given coordinate
'''
 def get_mandelbrot(c, max_iterations):
     
   """
   Compute mandelbrot set for a given coordinate
   """
   z = zeros(max_iterations)
   for i in range(max_iterations):
      z[i] = z[i-1]**2 + c
      if abs(z[i]) > 2:
                
'''