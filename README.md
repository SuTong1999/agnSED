# agnSED

Contact sutong@bao.ac.cn/aosagnai@gmail.com for questions and collaborations. 

This model is used to calculate the spectral energy distribution (SED) of BH accretion proccesses. The input parameters are BH mass and accretion rate. The accretion flow is divided into two regime based on the value of Eddington-normalized accretion rate - the ADAFs for low accretion rate objects and the modified magnetic reconnection-heated disk-corona model for high accretion rate objects.
Detailed explanation and application on simulated galaxy catalog please refer to Su et al. 2026. 

## Citation

If you use this package in your research and publications, please cite:

- **Tong Su**, *Modeling the Spectral Energy Distribution of Active Galactic Nuclei: Implications for Cosmological Simulations of Galaxy Formation*, DOI: [10.3847/1538-4357/ae41bd](https://doi.org/10.3847/1538-4357/ae41bd)
- GitHub Repository: [https://github.com/SuTong1999/agnSED](https://github.com/SuTong1999/agnSED)

## Example Usage

An example Jupyter Notebook demonstrating how to use this package is available in the github repository.
- [[View on GitHub]](https://github.com/SuTong1999/agnSED/tree/main/example)

## Command-line Install

To install this package into your local pip-enabled python environment,
```bash
git clone https://github.com/SuTong1999/agnSED
cd agnSED
pip install .
```

## Release Note
(2026/3/31)
Restored data/ directory.

(2025/2/1)
To avoid potential conflicts with our upcoming project, the data directory (located at src/agnSED/data/) has been temporarily removed. If you require access to this package, please contact me directly. I will be happy to provide the necessary files through private communication.

(2025/1/22)
Scipy 1.15.1 changed the internal attributes of the RegulatorGridInterpolator function, this will produce error message "AttributeError: 'RegularGridInterpolator' object has no attribute '_spline'" when executing example.ipynb. To avoid this, the install requirement in the setup.cfg constrained Scipy version to scipy>=1.10.0,<=1.14.1. This issue will be fixed in the (foreseeable) future. *Installing this package could automatically downgrade/upgrade your scipy (if you are using a higher/lower version), so the recommended method is to install it in a new python environment.*


