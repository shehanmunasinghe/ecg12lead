from setuptools import setup, find_packages

setup(
    name='ecg12lead',
    version='0.0.0',
    description='Deep-learning based classification of 12-lead ECG signals',
    author="shehanmunasinghe",
    author_email='shehanmunasinghe@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/shehanmunasinghe/ecg12lead",
    install_requires=['torch'],
    packages=find_packages(),
)