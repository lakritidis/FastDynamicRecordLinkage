from distutils.core import setup
from setuptools import find_packages

DESCRIPTION = 'Fast Dynamic Record Linkage'
LONG_DESCRIPTION = '<p>FaDReL is a library for entity matching/record linkage applications</p>'

setup(
    name='FaDReL',
    version='0.0.1',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author="Leonidas Akritidis",
    author_email="lakritidis@ihu.gr",
    maintainer="Leonidas Akritidis",
    maintainer_email="lakritidis@ihu.gr",
    packages=find_packages(),
    package_data={'': ['generators/*']},
    url='https://github.com/lakritidis/fadrel',
    install_requires=["numpy",
                      "pandas",
                      "torch>=2.0.0",
                      "scikit-learn>=1.4.0",
                      "transformers>=4.4.0",
                      "sentence_transformers>=0.8.0",
                      "tqdm>=4.60.0",],
    license="Apache",
    keywords=[
        "tabular data", "tabular data synthesis", "data engineering", "imbalanced data", "GAN", "VAE", "oversampling",
        "machine learning", "deep learning"]
)
