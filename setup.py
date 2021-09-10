from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.41'
DESCRIPTION = 'TMH Speech package'
LONG_DESCRIPTION = 'A package for TMH Speach'

# Setting up
setup(
    name="tmh",
    version=VERSION,
    author="Birger Moell",
    author_email="<bmoell@kth.se>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['transformers', 'torch', 'torchaudio', 'speechbrain',
                      'librosa', 'numpy', 'scipy', 'unidecode', 'inflect', 'librosa', 'python-dotenv'],
    keywords=['python', 'speech', 'voice'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
