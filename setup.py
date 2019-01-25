from setuptools import setup


setup(
    name='pengle',
    version='0.0.1',
    author='Ryoji Nagata',
    description='A library for kaggle.',
    packages=['pengle', 'pengle.dataset', 'pengle.storage', 'pengle.trainer', 'pengle.transformer']
)