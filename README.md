
# Electrocatalyst Discovery through Text Mining and Multi-Objective Optimization

This repository contains the code and workflows described in the paper *"Accelerating Electrocatalyst Discovery through Text Mining and Multi-Objective Optimization"*. The project uses natural language processing and multi-objective Pareto optimization to accelerate material discovery processes.

## Getting Started

### Prerequisites
- Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Clone this repository to your local machine.
- Ensure material systems are placed in the correct directories:
   - ORR and HER material systems:`02_pareto_prediction/material_systems/MinDMaxC`
   - OER material systems:`02_pareto_prediction/material_systems/MaxDMinC`
   - To get one of the plot (scatter plot of current density for NiPdPtRu system), put the subsets data into `02_pareto_prediction/material_systems/MinDMaxC/Ni_Pd_Pt_Ru_subset` directory.


### Setting Up the Environment
Use the `environment.yml` file to create the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate your_env_name
   ```
   Replace `your_env_name` with the environment name in `environment.yml`.

## Workflow Overview

### Word2Vec Model Creation

1. Navigate to the `01_word2vec_model` directory and update the `config.yaml` file:
   - Add your Scopus API key under the `APIKey` entry.
   - Adjust other settings if necessary.

2. Run the Snakefile to collect papers, preprocess them, and generate a Word2Vec model:
   ```bash
   snakemake -c1
   ```
   The generated model will be stored in the `./model` directory.

### Pareto Front Predictions
1. Navigate to the `01_pareto_prediction` directory.
2. Run the Snakefile:
   ```bash
   snakemake -c1
   ```
   The results will include Pareto front predictions for the specified material systems.

### Generating Plots
1. Navigate to the `03_plots` directory.
2. Run the Snakefile:
   ```bash
   snakemake -c1
   ```
   All plots will be generated and saved in the directory.

### Generating Tables
1. Navigate to the `04_tables` directory.
2. Run the Snakefile:
   ```bash
   snakemake -c1
   ```
   All tables will be generated and saved in the directory.

## License
This project is licensed under the LGPL-3.0 License. See the [LICENSE](LICENSE) file for details.
