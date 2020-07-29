"""TFF Aggregations is a lib for secure aggregation in TensorFlow Federated.

TFF Aggregations uses primitives from
[TF Encrypted](https://github.com/tf-encrypted/tf-encrypted) to define and
execute secure versions of federated aggregation functions from TFF's
Federated Core."""
import setuptools

DOCLINES = __doc__.split('\n')
REQUIRED_PACKAGES = [
    'tensorflow-federated>=0.16.0',
    'tf-encrypted-primitives>=0.1.0',
]

with open('federated_aggregations/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
  VERSION = globals_dict['__version__']

setuptools.setup(
    name='federated_aggregations',
    version=VERSION,
    packages=setuptools.find_packages(exclude=('examples')),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    long_description_content_type='text/markdown',
    author='The TF Encrypted Authors',
    author_email='contact@tf-encrypted.io',
    url='https://github.com/tf-encrypted/federated-aggregations',
    download_url='https://github.com/tf-encrypted/federated-aggregations/tags',
    install_requires=REQUIRED_PACKAGES,
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    license='Apache 2.0',
    keywords='tensorflow encrypted federated machine learning',
)

