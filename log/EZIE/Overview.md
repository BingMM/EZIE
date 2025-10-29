# EZIE - Auroral electrojet reconstruction

EZIE is a satellite mission that remotely senses the ambient magnetic field in the mesosphere (around 80 km altitude). Each of the three CubeSats, flying like pearls on a string, carries four MEM sensors. This results in four _measurement tracks_ of oxygen thermal emission, from which the ambient magnetic field is retrieved.

This repository contains the codebase for modeling the equivalent ionospheric electric current using EZIE magnetic field measurements. In other words, generating Level 3 (L3) data from Level 2 (L2) data.

The code is designed to be modular and makes use of lazy loading. It currently includes the following classes:

- **Data** – Stores and manages the `MEM` class.

- **MEM** – Stores data for each MEM instrument.

- **Model** – Takes the `Data` class as input and retrieves the equivalent current, leveraging the `RegularizationOptimization` class if regularization parameters are not manually set.

- **RegularizationOptimizer** – Determines the optimal set of regularization parameters.

- **Plotter**

- **Evaluator**
