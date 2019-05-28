import os

from setuptools import setup, find_packages

setup_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(setup_dir, 'README.md')) as f:
    readme = f.read()

with open(os.path.join(setup_dir, 'LICENSE')) as f:
    lic = f.read()

with open(os.path.join(setup_dir, 'requirements.txt')) as f:
    reqs = f.read().splitlines()

setup(
    name='param',
    version='0.0.0',
    description='tba',
    long_description=readme,
    author_email='tba',
    author='tba',
    url='tba',
    license=lic,
    packages=find_packages(),
    install_requires=reqs
)
