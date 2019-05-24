from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    lic = f.read()

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(
    name='param_est',
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
