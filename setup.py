from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='param_est',
    version='0.0.0',
    description='tba',
    long_description=readme,
    author='tba',
    author_email='tba',
    url='tba',
    license=license,
    packages=find_packages()
)
