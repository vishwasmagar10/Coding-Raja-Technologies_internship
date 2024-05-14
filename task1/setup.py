from setuptools import setup

with open("README.md","r",encoding="utf-8") as fh:
    long_description = fh.read()

AUTHOR_NAME = 'VISHWAS'
SRC_REPO = 'src'
LIST_OF_REQUIREMENTS = ['streamlit']

setup(
    name = SRC_REPO,
    version= '0.0.1',
    author = AUTHOR_NAME,
    author_email= 'vishwasmagar10@gmail.com',
    description = 'A small example package for movie recommendation',
    long_description=long_description,
    long_description_content_type= 'text/markdown',
    packages = [SRC_REPO],
    python_requires = '>=3.7',
    install_requires = LIST_OF_REQUIREMENTS,
)