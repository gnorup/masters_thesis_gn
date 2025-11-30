# Spontaneous Speech as a Marker of Language Cognition in Healthy Aging: Regression and Classification Approaches

## Overview
This project contains code for all analyses conducted as part of the master's thesis "Spontaneous Speech as a Marker of Language Cognition in Healthy Aging: Regression and Classification Approaches".  
Author: Gila Norup  
Supervisors: Jonathan Heitz & Nicolas Langer  
Chair: Methods of Plasticity Research, University of Zurich 

## Project Structure
- `data`: Demographic data and language scores for sample; pre-stratified folds for cross-validation.
- `env_setup`: Conda environment file
- `resources`: Lexicons required for calculating linguistic features, selection of manually calculated scores for score validation 
- `results`: Main results presented in thesis.
  - `results/classification`: Classification results for all scores and models 
  - `results/correlations`: Correlation results for language scores, features and demographics 
  - `results/data_preparation`: Details on data cleaning process 
  - `results/dataset`: Descriptives, distributions 
  - `results/features`: Raw extracted features and cleaned feature sets for analyses
  - `results/regression`: Regression results for Random Forest and multiple linear regression, including performance metrics, comparisons of scores, tasks and models
  - `results/regression/random_forest/`: Post-Analyses for Random Forest regression
    - `bias`: Predictive performance across demographic subgroups
    - `feature_importance`: SHAP values for all scores and tasks
    - `picture_description`: Picture description sample length analyses 
- `src`: Main directory for source code
  - `src/additional_analyses`: Correlation, score validation, dataset descriptives and Jupyter notebooks
  - `src/classification`: Classification analyses 
  - `src/config`: YAML file to select which analyses to run in the main pipeline, constants and feature sets
  - `src/data_preparation`: Feature set cleaning, stratified splits 
  - `src/feature_extraction`: Feature extraction, linguistic and acoustic features 
  - `src/regression`: Main regression analyses and post analyses for Random Forest regression 
  - `run_all_analyses.py`: Main pipeline script; run all selected analysis steps 

## Data
The project uses data from the _Language Healthy Aging_ dataset collected by Heitz et al. (2025). 

## Setup
1. Create conda environment from the environment file (`env_setup/environment.yml`)
2. Change paths in `src/config/constants.py`
3. Select which analyses to run in `src/config/config.yaml`
4. Run pipeline from `src/run_all_analyses.py`
