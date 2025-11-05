# EZIE Auroral Electrojet modeling code

`EZIE` is a tool for processing EZIE L2 data to estimate the equivalent ionospheric electric current using SECS.

> ⚠️ `EZIE` is not an official repository for generating L3 EZIE data.  

## Project Description

This code was developed to robustly estimate the equivalent ionospheric electric current from EZIE data.

## Dependencies

- [`secsy`](https://github.com/klaundal/secsy) - for cubed sphere grid generation and SEC basis functions.
- [`tqdm`](https://github.com/tqdm/tqdm) – for progress bars (optional; can be removed with minor edits).

## Installation

mamba activate your_environment  
git clone https://github.com/BingMM/EZIE.git  
cd EZIE  
pip install -e .

## User guide

See test script in the script subfolder

## References

**[1]**. Laundal, K. M., et al. (2021). Electrojet estimates from mesospheric magnetic field measurements. Journal of Geophysical Research: Space Physics, 126, e2020JA028644. https://doi.org/10.1029/2020JA028644
**[2]**.  Madelaire, M., et al. (2023). Spatial resolution in inverse problems: The EZIE satellite mission. Journal of Geophysical Research: Space Physics, 128, e2023JA031394. https://doi.org/10.1029/2023JA031394

## License

This project is licensed under the BSD-3 License. See the `LICENSE` file for details.

## Acknowledgments
This project includes a modified ['SuperMAG API python client'](https://supermag.jhuapl.edu/mag/?tab=api)

## Contact

For questions or comments, please contact [micmad@dtu.dk].
