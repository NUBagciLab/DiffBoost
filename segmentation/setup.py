from setuptools import setup, find_packages

print(find_packages())

setup(
    name = 'segmentation',
    version="v1.0",
    description="MedSegAug",
    author="zheyuan",
    author_email="zhangzheyuan14@gmail.com",
    license="MIT",
    packages = find_packages(),
)