# Embedding Riemannian Collective Background Knowledge for Offline Signature Verification

This repository contains the official MATLAB implementation for the evaluation, demonstration, and reproducibility of the frameworks presented in the paper:

> **Embedding Riemannian Collective Background Knowledge for Offline Signature Verification**  
> *Mitikas, E., Chorianopoulos, C., & Zois, E. (2026).* Machine Learning and Knowledge Extraction, 8(7), 208.  
> [https://doi.org/10.3390/make8070208](https://doi.org/10.3390/make8070208)

---

## Overview

This software implements an end-to-end Riemannian framework featuring the **Collective Background Knowledge (CBK)** mechanism for offline signature verification. Moving beyond conventional Euclidean assumptions, signatures are modeled as points on the Symmetric Positive Definite (SPD) manifold. 

The framework establishes synthetic writers as Riemannian centers and utilizes a learnable $\alpha\beta$-Log-Determinant divergence to adaptively discover the optimal local geometry from data. Instead of computing direct distances between two signature representations, the pipeline evaluates them relationally by tracking how each signature diverges from the shared CBK reference centers. These deviations form unique relational profiles, which are then compared using the dichotomy transform to create dissimilarity vectors.

This repository specifically reproduces the experimental results for the **L5** and **L5G** scenarios using the **CEDAR** dataset within an **$F_{IL}$** setting.

## Running the Framework

Execute the main script to run the $5 \times 2$ cross-validation fold iterations for either the L5 or L5G scenario.
* Open MATLAB.
* Run the following command in the Command Window:
```matlab
sign_model  # Run the framework
result_plot(models, "CEDAR")    # Plot final Error Rate (EER%)
```

## Copyright & Licensing

Please refer to the LICENSE file for the Copyright & Licensing disclosure.
