"""

  plot_MMM_speedups_ssweep.py - A Python script that uses NumPy and Matplotlib to
                                plot the speedup (according to Amdahl's and
                                Gustafson's Laws) for MatMatMult.py

  Notes:                        The number of processors, p, is fixed and is
                                equal to 4

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# set the number of processors using a logarithmic, base 2, sequence from 2**startExp to 2**endExp
startExp = 2
endExp = 11
points = endExp - startExp + 1
x = np.logspace ( startExp, endExp, points, base = 2 )

# set labels for x axis (matrix size)
labels = ['4x4', '8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512', '1024x1024', '2048x2048']
# speedups according to Amdahl's Law
yA = [2.79, 2.71, 3.30, 3.65, 3.83, 3.90, 3.96, 3.98, 3.99, 3.99]
# speedups according to Gustafson's Law
yG = [3.56, 3.53, 3.79, 3.90, 3.96, 3.97, 3.99, 3.99, 4.00, 4.00]

# initialise the figure and axes
fig, ax = plt.subplots ()
# set the title
fig.suptitle ( "Speedups for MatMatMult.py", fontweight = "bold" )
# set axes names
plt.xlabel ( "matrix size" )
plt.ylabel ( "$\psi$" )
plt.gcf().subplots_adjust ( bottom = 0.2 )
# set the x scale as logarithmigc, base 2
ax.set_xscale ( 'log', base = 2 )
# use rotated labels (30 degrees) as ticks in the x axis
plt.xticks ( x, labels )
plt.xticks ( rotation = 30 )

# plot using a grid and red/orange circle markers
plt.grid ()
plt.plot ( x, yA, color = 'red', marker = 'o', label = "Amdahl's Law" )
plt.plot ( x, yG, color = 'orange', marker = 'o', label = "Gustafson's Law" )
# put the legend in the lower right corner
ax.legend ( loc = 'lower right' )

plt.savefig ( "MMM_speedups_ssweep.png" )
plt.show ()
