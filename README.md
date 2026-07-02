# srimpro
srimpro is a python library that contains many tools for the automated extracting, calculating, plotting, and/or exporting of SRIM data for a wide range of purposes. SRIM is a popular software package used to simulate the irradiation of materials by energetic ions, but extracting data from its output files, performing calculations, and plotting data can be extremely time consuming. By automating these processes, srimpro aims to improve the speed and efficiency of SRIM data analysis. Additionally, srimpro contains advanced functions that enable SRIM to be used for a wider range of purposes than what has traditionally been available to researchers.
* srimpro reads data from SRIM output files using an updated version of [pysrim](https://github.com/costrouc/pysrim)
* all functions of srimpro are fully documented, and all code is fully commented

# Installation
srimpro can easily be installed using the following command in Anaconda Prompt or similar application:
```
pip install git+https://github.com/Noah-Miggs/srimpro.git
```

# Setup
After installing srimpro, all dependencies including pysrim should be automatically installed as well. srimpro requires updated versions of the **output.py**, **srim.py**, and **elementdb.py** files within pysrim to make it compatible with python 3.14, so the original files from pysrim must be manually replaced. See **srimpro Setup** in documentation for more details.
