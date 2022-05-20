#*******************************************************************************
#
#  setup.py - A Python script to compile the Cython code
#
#   Notes:    Cython is is an extension of Pyrex that contains several enhancements
#
#*******************************************************************************

from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension ( name = "xsqax", sources = ["xsqax.pyx"] )
setup ( ext_modules = cythonize ( ext, language_level = "3" ) )
