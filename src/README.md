# Source
This is the main folder. It contains all classes and routines that are called in the main script "run_exp.py".

- datasets:
    - BaseDataset.py: (not really used) abstract class for formatting asset-level datasets. Useful for duplicating experiments with several sectors.
    - CEMNETDataset.py: CEMNET dataset
    - EmissionFactors: class used to manipulate emission factors from several sources and define an imputation strategy.
    - GSPTDataset.py: Global Steel Plant Tracker 22 interface. Access steel plant data in various format, with data imputation.
    - RefinitivDataset.py: Refinitiv interface.
    - utils.py: helper functions for processing datasets (e.g. cleaning ownership data...).
- BUProjector.py: class used for projecting emissions with two different methods (top-down and bottom-up). This class has method for calculating and plotting prediction intervals of projections.
- CI.py: helpers for calculating linear regression prediction intervals.
- cross_val.py: helpers for performing cross validation.
- feature_engineering.py: helpers for integrating new features to the asset-level dataset.
- match_names.py: standalone script for matching company names from two different columns.
