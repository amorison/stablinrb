import os

from setuptools import setup

with open('README.rst') as rdm:
    README = rdm.read()

DEPENDENCIES = [
    'dmsuite>=0.1.1',
    'setuptools_scm>=1.15',
]
HEAVY = [
    'numpy>=1.12',
    'scipy>=1.0',
    'matplotlib>=3.0',
]

ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'
# heavy dependencies are mocked out on Read the Docs
if not ON_RTD:
    DEPENDENCIES.extend(HEAVY)

setup(
    name='stablinrb',
    use_scm_version=True,

    description='Rayleigh-Bénard linear stability analysis',
    long_description=README,

    url='https://github.com/amorison/stablinrb',

    author='Adrien Morison, Stéphane Labrosse',
    author_email='adrien.morison@gmail.com',

    license='Apache',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    python_requires='>=3.5',
    packages=['stablinrb'],
    install_requires=DEPENDENCIES,
)
