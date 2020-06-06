import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reed-wsd", # Replace with your own username
    version="0.0.1",
    author="Mark Hopkins",
    author_email="hopkinsm@reed.edu",
    description="A package for word sense disambiguation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mark-Hopkins-at-Reed/research-wsd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)