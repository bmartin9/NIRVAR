from setuptools import find_packages, setup

with open("README.md", "r") as f:
    description = f.read()

setup(
    name='nirvar',
    packages=find_packages(),
    version='1.0.0',
    description='Network Informed Restricted Vector Autoregression',
    author='Brendan Martin',
    license='MIT',
    long_description=description,
    long_description_content_type="text/markdown",
)