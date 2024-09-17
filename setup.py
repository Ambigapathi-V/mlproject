from setuptools import find_packages,setup



setup(
    name='mlproject',
    version='0.0.1',
    author='Ambigapathi',
    author_email='ambigapathikavin@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
    ],
)