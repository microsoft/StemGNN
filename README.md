# Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting

This repository is the official implementation of Spectral Temporal Graph Neural Network for
Multivariate Time-series Forecasting.

<!-- > 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

<!-- > 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->

## Training and Evaluation

The training procedure and evaluation procedure are all included in the `main.py`. To train and evaluate on some dataset, run the following command:

```train & evaluate
python main.py --dataset <path_to_data>
```

<!-- The hyperparameters are set as default pramaters for reproduction convenience, so no more parameters need to be specified in the above command. -->

## Datasets

| Dataset | Hyperparameters |
| ------- | --------------- |
| [METR-LA](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit) | `xxx` |
| [PEMS-BAY](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit) |  |
| [PEMS03](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit) |  |
| [PEMS04](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit) |  |
| [PEMS07](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit) |  |
| [PEMS08](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit) |  |
| [Solar](https://www.nrel.gov/grid/solar-power-data.html) |  |
| [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) |  |
| [ECG](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000) |  |

Since some preprocessings need to be conducted on the above datasets provided in the urls before fed into StemGNN model, we provide a preprocessed version of XXXX (relative path to XXXX) for reproduction convenience.

## Results

Our model achieves the following performance on the 3 datasets included in the code repo:

| Evaluation Method | Dataset 1  | Dataset 2 | Dataset 3 |
| ------------------ |---------------- | -------------- |  -------------- |
| MAE  | xxx | xxx | xxx |
| RMSE | xxx | xxx | xxx |
| MAPE | xxx | xxx | xxx |

<!-- > 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.  -->

