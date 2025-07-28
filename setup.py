from setuptools import setup, find_packages

setup(
    name="pylabelme2yolo",
    version="0.1.2",
    author="shyandsy",
    author_email="shyandsy@gmail.com",
    description="Convert LabelMe annotations to YOLO format",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shyandsy/pylabelme2yolo",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "typer>=0.4.0",
        "Pillow>=8.0.0",
        "tqdm>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "pylabelme2yolo=pylabelme2yolo.cli:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)