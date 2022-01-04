import numpy

size = 256
spokes = 360

spoke_range = (numpy.arange(0, size) - 128.0 )* numpy.pi/ (size/2)  # normalized between -pi and pi
M = size*spokes
om = numpy.empty((M,2), dtype = numpy.float32)


for angle in range(0, spokes):
   radian = angle * 2 * numpy.pi/ spokes
   spoke_x =  spoke_range * numpy.cos(radian)
   spoke_y =  spoke_range * numpy.sin(radian)
   om[size*angle : size*(angle + 1) ,0] = spoke_x
   om[size*angle : size*(angle + 1) ,1] = spoke_y


import matplotlib.pyplot
matplotlib.pyplot.plot(om[:,0], om[:,1],'.')
matplotlib.pyplot.show()