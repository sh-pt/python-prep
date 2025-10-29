from setuptools import setup, Extension
from Cython.Build import cythonize

exts = [
    Extension(
        name="stream_core",
        sources=['stream_core.pyx'],
    )
]

setup(
    name='q3_moving_average',
    version='0.0.1',
    ext_modules=cythonize(
        exts,
        language_level='3',
        annotate=True,
    ),
)