from setuptools import find_packages, setup


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


with open('README.md') as f:
    readme = f.read()


install_requires = parse_requirements('requirements.txt')
setup(
    name='word_embeddings_sdk',
    packages=find_packages(),
    version='0.1.0',
    license='MIT',
    description='Python sdk to interface with the WordEmbeddings API',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Patrick Hennis',
    author_email='patrick@width.ai',
    url='https://github.com/Width-ai/embeddings-sdk',
    download_url='https://github.com/Width-ai/embeddings-sdk/archive/refs/tags/v0.1.0.tar.gz',
    keywords=['Embeddings', 'SDK', 'WordEmbeddings', 'WordEmbeddings.Ai'],
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)