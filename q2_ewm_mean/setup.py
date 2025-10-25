from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

exts = [
    Extension(
        "ewmcore._ewm_fast",
        sources=["src/ewmcore/_ewm_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=cythonize(
        exts,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
        },
    ),
    package_dir={"": "src"},
    packages=["ewmcore"],
)
