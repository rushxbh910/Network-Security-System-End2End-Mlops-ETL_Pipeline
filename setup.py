'''
The setup.py file is an essential part of packaging and sidtributing python projects.
It is used by setuptools to define the configuration of your project, such as its metadeta, dependencies, and more
'''

from setuptools import find_packages,setup
from typing import List

def  get_requirements()->List[str]:
    '''
    This function will return a list of requirements
    '''
    requirements_list:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #read lines from the file
            lines=file.readlines()

            for line in lines:
                requirement=line.strip()
                if requirement and requirement !='-e .':
                    requirements_list.append(requirement)
    
    except FileNotFoundError:
        print("requirements.txt file not found!")

    return requirements_list

print(get_requirements())

setup(
    name="Network-Security-System-End2End-Mlops-ETL_Pipeline",
    version="0.0.1",
    author="Rushabh Bhatt",
    author_email="rushxbh910@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)