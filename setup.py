from setuptools import setup
from setuptools import find_packages

long_description = '''
Pwml is a high-level machine learning API, written in Python.
Pwnl stands for Python Wrappers for Machine Learning.
Pwml is compatible with Python 3.6 and is distributed under the MIT license.
'''

setup(
    name='Pwml',
    version='0.9.8',
    description='Python Wrappers for Machine Learning',
    long_description=long_description,
    author='Benjamin Raibaud',
    author_email='braibaud@gmail.com',
    url='https://github.com/braibaud/pwml',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires=[
        'numpy>=1.18.0',
        'pillow>=5.0.0',
        'pymssql>=2.0.0',
        'pandas>=1.0.0',
        'urllib3>=1.25.0',
        'scikit-learn>=0.22.0',
        'scipy>=1.4.0',
        'tensorflow-hub>=0.6.0',
        'tensorflow>=2.0.0'],
    python_requires='>=3.6',
    packages=find_packages())