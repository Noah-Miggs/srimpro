# srimpro
srimpro is a python library that contains many tools for the automated extracting, calculating, publication-quality plotting, and/or exporting of SRIM data for a wide range of purposes. SRIM is a popular software package used to simulate the irradiation of materials by energetic ions, but extracting data from its output files, performing calculations, and plotting data can be extremely time consuming. By automating these processes, srimpro aims to improve the speed and efficiency of SRIM data analysis. Additionally, srimpro contains advanced functions that enable SRIM to be used for a wider range of purposes than what has traditionally been available to researchers.
* srimpro reads data from SRIM output files using an updated version of [pysrim](https://github.com/costrouc/pysrim)
* most functions have several optional inputs to provide a wide range of utilities
* plot settings can be easily modified within each function to allow for full ploot customization
* all functions are fully documented, and all code is fully commented

# Features
srimpro is designed to have a high ease of use. Most functions only require the `path` input (filepath to the folder containing the appropriate SRIM output files) to generate an output.
## Integration Functions
These functions are used to integrate under the profiles from SRIM to get total values. For example, the following code will set *vactot* and *novactot* to the total number of vacancies and replacements between depth x1 and x2 respectively:
```
vactot, novactot = totDisplacements(path, x1, x2)
```

## Basic SRIM Plots
These functions are used to generate publication-quality plots or export data to excel, specfically for the data commonly used from SRIM. These include plotting ion distribution and damage dose, energy deposition, and stopping power. For example, the following code will generate the plot shown below:
```
rangeAndDpa(path)
```
![1.5 MeV Si in ZrN (1000 nm) Si (500 nm), 16,000 ions calculated][(https://github.com/Noah-Miggs/srimpro](https://github.com/Noah-Miggs/srimpro/tree/main/example_plots/1.5-MeV-Si-in-ZrN-and-Si-io-distribution-and-damage-dose.png)

# Installation
srimpro can easily be installed using the following command in Anaconda Prompt or similar application:
```
pip install git+https://github.com/Noah-Miggs/srimpro.git
```

# Setup
After installing srimpro, all dependencies including pysrim should be automatically installed as well. srimpro requires updated versions of the *output.py*, *srim.py*, and *elementdb.py* files within pysrim to make it compatible with python 3.14, so the original files from pysrim must be manually replaced. See *srimpro Setup* in documentation for more details.
