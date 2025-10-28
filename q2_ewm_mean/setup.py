from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

exts = [
    Extension(
        "ewmcore._ewm_fast",
        sources=["src/ewmcore/_ewm_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        define_macros=[("CYTHON_TRACE", "1")]
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
            "linetrace": True,
            "profile": True,
            "binding": True,
        },
    ),
    package_dir={"": "src"},
    packages=["ewmcore"],
)
