from setuptools import setup, find_packages

setup(
    name='wide_embeddings_sdk',
    version='0.1',
    description='SDK for Width.Ai Embeddings Service',
    url='https://github.com/Width-ai/',
    author='pat',
    author_email='patrick@width.ai',
    license='All Rights Reserved',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']), 
    install_requires=['requests'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="Embeddings API SDK",
)