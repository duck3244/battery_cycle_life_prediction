# Dataset Attribution and License

This repository contains **code only**. When you run the pipeline with
`--use-real-data`, it downloads a preprocessed subset of an external dataset
that is **not bundled** in this repository. The license and attribution
requirements below apply to that external data, not to the source code in
this repository.

## Upstream dataset

- **Title**: Data-driven prediction of battery cycle life before capacity
  degradation (experimental dataset).
- **Authors**: Kristen A. Severson, Peter M. Attia, Norman Jin, Nicholas
  Perkins, Benben Jiang, Zi Yang, Michael H. Chen, Muratahan Aykol, Patrick
  K. Herring, Dimitrios Fraggedakis, Martin Z. Bazant, Stephen J. Harris,
  William C. Chueh, Richard D. Braatz.
- **Publication**: *Nature Energy* **4**, 383–391 (2019).
  [doi:10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)
- **Host**: Toyota Research Institute, Experimental Data Platform
  ([data.matr.io/1/projects/5c48dd2bc625d700019f3204](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204))
- **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

## Redistribution used by this project

The pipeline downloads a preprocessed zip published by MathWorks as a support
file for their MATLAB example
*"Battery Cycle Life Prediction Using Deep Learning"*:

- URL: `https://ssd.mathworks.com/supportfiles/predmaint/batterycyclelifeprediction/v2/batteryDischargeData.zip`
- Archive entry: `batteryDischargeData.mat`

This is a discharge-only subset of the Severson et al. 2019 data, restricted
to the discharge portion of each cycle (voltage range 3.6 V → 2.0 V).

## Modifications performed by this project

When the pipeline runs, it performs the following transformations on the
downloaded data **in memory**. No modified data is redistributed by this
repository:

1. Extracts the discharge portion of each cycle using the 3.6 V → 2.0 V
   voltage window (`data_preprocessor.extract_discharge_data`).
2. Applies a length-3 uniform smoothing filter to V, T, and Qd.
3. Linearly interpolates each cycle to 900 points on the voltage axis and
   reshapes to a 30 × 30 × 3 tensor.
4. Applies MinMax (default) or z-score normalization; parameters fit on the
   training split only and saved to `backend/models/norm_params.npz`.

## Attribution requirement (CC BY 4.0)

Any derivative work — papers, blog posts, shared models, or screenshots of
plots produced by this project — that uses the Severson et al. data must
cite the paper above and indicate the CC BY 4.0 license. Retain this file
and the "Dataset Attribution & License" section of the `README.md` when
redistributing the code.

## Source code license

The source code in this repository is independent of the dataset license.
If no repository-level `LICENSE` file is present, no license is granted for
the code itself; treat it as "all rights reserved" until the author adds
an explicit source-code license.
