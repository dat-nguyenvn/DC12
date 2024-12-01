from setuptools import setup, find_packages

setup(
    name="wildtracker",                      # Your project name
    version="0.1.0",                        # Initial version
    author="Ngoc Dat Nguyen",
    author_email="dat.nguyenngoc@bristol.ac.uk",
    description="A brief description of your project",
    long_description=open("README.md").read(),  # Description from README
    long_description_content_type="text/markdown",
    url="https://github.com/dat-nguyenvn/DC12",  # Project's GitHub URL
    packages=find_packages(),               # Automatically find packages
    install_requires=[                      # Dependencies
        "numpy",
        "requests"
    ],
    classifiers=[                           # Optional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",                # Minimum Python version
)
