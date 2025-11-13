# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

exts = [
    Extension(
        name="ewm_fast",
        sources=["ewm_fast.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="q2_ewm_mean",
    version="0.0.1",
    py_modules=["src"],
    ext_modules=cythonize(
        exts,
        language_level="3",
        annotate=True,
    ),
)
