from setuptools.command import install
from setuptools import setup
from os import path
import io

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with io.open(path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(
  name='futhark_svm',
  version='0.0.1',
  description='A support vector machine implementation in Futhark',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/fzzle/futhark-svm',
  author='fzzle',
  packages=['futhark_svm'],
  setup_requires=['cffi'],
  cffi_modules=['build.py:ffi'],
  install_requires=[
    'numpy',
    'futhark_ffi',
    'futhark_data'
  ]
)