from setuptools import setup, find_packages
import os

# pip install wasnt finding al the submodules so I needed to do the find_packages function
# any subdirectoies that we dont want excluded need to be excluded via: exclude=


# also need to include all subdirs in the data directory
def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


extra_files = package_files("./nitrates/data")

try:
    with open("README.md", "r") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

with open("requirements.txt") as f:
    required = f.read().splitlines()

#get the version number from the _version file
with open("nitrates/_version.py") as f:
    file_info = f.read().splitlines()
version=file_info[-1].split("=")[-1].split('"')[1]


setup(
    name="nitrates",
    version=version,
    packages=find_packages(),
    url="https://github.com/Swift-BAT/NITRATES",
    license="BSD-3-Clause",
    author="Jimmy DeLaunay",
    author_email="delauj2@gmail.com",
    description="Analysis of GUANO-ed data from the Neil Gehrels Swift Observatory Burst Alert Telescope",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    # package_data={'nitrates': ['data/*']},
    package_data={"": extra_files},
    python_requires=">=3.8",
)
