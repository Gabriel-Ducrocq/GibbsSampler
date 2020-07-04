from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["variance_expension.pyx", "linear_algebra.pyx"]), include_dirs=[numpy.get_include()] ,annotate=True)
