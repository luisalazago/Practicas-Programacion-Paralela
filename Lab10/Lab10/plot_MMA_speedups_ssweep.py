"""

  plot_MMA_speedups_ssweep.py - A Python script that uses NumPy and Matplotlib to
                                plot the speedup (according to Amdahl's and
                                Gustafson's Laws) for MatMatAdd.py

  Notes:                        The number of processors, p, is fixed and is
                                equal to 4

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# set the number of processors using a logarithmic, base 2, sequence from 2**startExp to 2**endExp
startExp = 2
endExp = 9
xticks = endExp - startExp + 1
x = np.logspace ( startExp, endExp, xticks, base = 2 )

# set labels for x axis (matrix size)
labels = ['4x4', '8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512']
# speedups according to Amdahl's Law
yA = [1.35, 1.44, 1.47, 1.69, 1.60, 1.60, 1.56, 1.64]
# speedups according to Gustafson's Law
yG = [2.04, 2.23, 2.29, 2.63, 2.51, 2.49, 2.44, 2.56]

# initialise the figure and axes
fig, ax = plt.subplots ()
# set the title
fig.suptitle ( "Speedups for MatMatAdd.py", fontweight = "bold" )
# set axes names
plt.xlabel ( "matrix size" )
plt.ylabel ( "$\psi$" )
plt.gcf().subplots_adjust ( bottom = 0.15 )
# set the x scale as logarithmic, base 2
ax.set_xscale ( 'log', base = 2 )
# use rotated labels (30 degrees) in the x axis
plt.xticks ( x, labels )
plt.xticks ( rotation = 30 )

# plot using a grid and red/orange circle markers
plt.grid ()
plt.plot ( x, yA, color = 'red', marker = 'o', label = "Amdahl's Law" )
plt.plot ( x, yG, color = 'orange', marker = 'o', label = "Gustafson's Law" )
# put the legend in the lower right corner
ax.legend ( loc = 'lower right' )

plt.savefig ( "MMA_speedups_ssweep.png" )
plt.show ()
