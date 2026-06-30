# Creating simnext wheel package
# python setup.py bdist_wheel
#
# Installing  module  
# pip install file_name.whl
#
# Installing modeul in edit mode (development mode)
# pip install --force-reinstall -e path_of_setup.py_folder
#
# Showing installed sim module info
# pip show package_name
#
# Uninstalling sim module
# pip uninstall -y package_name

from setuptools import setup, find_packages

setup(
    name="grx",
    version="1.0.3",
    package_dir={'': 'source'},                         # consider the folder as source (othewise the package will be source.sim instaed of sim). 
    packages=['grx'],           # contains sim functionalities only.  
    package_data={'' : ['data/**/*'],},                 # contains sim data    
    #packages=find_packages(where="source"),
    #include_package_data=True,
    #package_data={"simnext": ["data/*", "data/**/*"]},
    python_requires=">=3.10",
)