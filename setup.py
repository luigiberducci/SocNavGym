import os
import sys
from setuptools import find_packages, setup


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "socnavgym"))

NAME = "socnavgym"
DESCRIPTION = "An environment for Social Navigation"
REPOSITORY = "https://github.com/gnns4hri/SocNavGym"
EMAIL = "sushantswamy1212@gmail.com"
AUTHOR = "Sushant Swamy"
VERSION = "0.1.0"

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

REQUIRED = [
    "gymnasium",
    "opencv-python",
    "numpy",
    "matplotlib",
    "shapely",
    "Cython",
    "pyyaml",
    "pandas",
    # learning
    "torch",
    "stable-baselines3",
    "shimmy>=0.2.1", # use gym env in sb3
    # for fosco
    "aim",
    "pygame",
    "moviepy",
    # for clean-rl
    "cvxpylayers",
    "tensorboard",
    "tyro",
]
EXCLUDES=["environment_configs", "tests"]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=REPOSITORY,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=EXCLUDES),
    install_requires=REQUIRED,
    license="GPL-3",
    include_package_data=True
)
