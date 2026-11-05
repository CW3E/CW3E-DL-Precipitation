# CW3E-DL-Precipitation

This repository contains Python code for the deep learning precipitation post-processing framework discussed in the following manuscript:

Yu, S., Sengupta, A., Dixon, T., Haleakala, K., Chen, A., & Delle Monache, L. (2026). *Deep Learning Post-Processing for Calibrated and Spatiotemporally Coherent Precipitation Ensembles*. Manuscript in preparation.

Center for Western Weather and Water Extremes, Scripps Institution of Oceanography, University of California, San Diego, La Jolla, California.

## Contents

1. **`1.DL_train.py`** — Deep learning calibration of precipitation forecasts using a censored shifted gamma distribution (CSGD) and cross-validation.
2. **`2.Schaake_Shuffle.py`** — Generation of 200-member precipitation ensembles using the Schaake Shuffle.
3. **`3.evaluation.py`** — Visualization and evaluation.

## Environment

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- NetCDF4, GeoPandas, Cartopy

