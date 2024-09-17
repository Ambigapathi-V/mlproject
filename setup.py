from setuptools import find_packages, setup
from typing import List

# Define the string '-e .' which may be present in requirements.txt
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        # Remove '-e .' if it's in the requirements
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Ambigapathi',
    author_email='ambigapathikavin@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
