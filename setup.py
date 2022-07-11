from setuptools import setup

try:
    with open("README.md", 'r') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''

setup(
    name='nitrates',
    version='0.1a1',
    packages=['nitrates'],
    url='https://github.com/Swift-BAT/NITRATES',
    license='BSD-3-Clause',
    author='Jimmy DeLaunay',
    author_email='delauj2@gmail.com',
    description='Analysis of GUANO-ed data from the Neil Gehrels Swift Observatory Burst Alert Telescope',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires = ['astropy', 'bs4', 'healpy', 'matplotlib', 'numpy', 'pandas', 'requests', 'scipy'],
    classifiers=['Development Status :: 3 - Alpha', 'Intended Audience :: Science/Research', 'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering :: Astronomy', ],
    python_requires='>=2.7',
)