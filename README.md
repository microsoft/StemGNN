# Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting

This repository is the official implementation of Spectral Temporal Graph Neural Network for
Multivariate Time-series Forecasting.

## Requirements

Recommended version of OS & Python:

* **OS**: Ubuntu 18.04.2 LTS
* **Python**: python3.7 ([instructions to install python3.7](https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/)).

To install python dependencies, virtualenv is recommended, `sudo apt install python3.7-venv` to install virtualenv for python3.7. All the python dependencies are verified for `pip==20.1.1` and `setuptools==41.2.0`. Run the following commands to create a venv and install python dependencies:

```setup
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```



## Datasets


[PEMS03](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit),
[PEMS04](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=4&submit=Submit),
[PEMS07](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=7&submit=Submit),
[PEMS08](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=8&submit=Submit),
[METR-LA](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit),
[PEMS-BAY](http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit),
[Solar](https://www.nrel.gov/grid/solar-power-data.html),
[Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014),
[ECG](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)

Since complex data cleansing is needed on the above datasets provided in the urls before fed into the StemGNN model, we provide a cleaned version of ECG ([./dataset/ECG_data.csv](./dataset/ECG_data.csv)) for reproduction convenience. The ECG_data.csv is in shape of `T*N`, where `T` denotes total number of timestamps, `N` denotes number of nodes. Run command `python main.py` to trigger training and evaluation on ECG_data.csv.

## Training and Evaluation

The training procedure and evaluation procedure are all included in the `main.py`. To train and evaluate on some dataset, run the following command:

For PEMS03 dataset:

```train & evaluate PEMS03
python main.py --train True --evaluate True --dataset ./dataset/PEMS03.csv --output_dir ./output/PEMS03 --n_route 358 --n_his 12 --n_pred 12
```

For PEMS04 dataset:

```train & evaluate PEMS04
python main.py --train True --evaluate True --dataset ./dataset/PEMS04.csv --output_dir ./output/PEMS04 --n_route 307 --n_his 12 --n_pred 12
```

For PEMS08 dataset:

```train & evaluate PEMS08
python main.py --train True --evaluate True --dataset ./dataset/PEMS08.csv --output_dir ./output/PEMS08 --n_route 170 --n_his 12 --n_pred 12
```

For PEMS04 dataset:

```train & evaluate PEMS04
python main.py --train True --evaluate True --dataset ./dataset/PEMS04.csv --output_dir ./output/PEMS04 --n_route 228 --n_his 12 --n_pred 3
```

For METR-LA dataset:

```train & evaluate METR-LA
python main.py --train True --evaluate True --dataset ./dataset/METR-LA.csv --output_dir ./output/METR-LA --n_route 207 --n_his 12 --n_pred 3
```

For PEMS-BAY dataset:

```train & evaluate METR-LA
python main.py --train True --evaluate True --dataset ./dataset/PEMS-BAY.csv --output_dir ./output/PEMS-BAY --n_route 325 --n_his 12 --n_pred 3
```

For Solar dataset:

```train & evaluate Solar
python main.py --train True --evaluate True --dataset ./dataset/Solar.csv --output_dir ./output/Solar --n_route 137 --n_his 12 --n_pred 3
```

For Electricity dataset:

```train & evaluate Electricity
python main.py --train True --evaluate True --dataset ./dataset/Electricity.csv --output_dir ./output/Electricity --n_route 321 --n_his 12 --n_pred 3
```

For ECG dataset:

```train & evaluate ECG
python main.py 
```


## Results

Our model achieves the following performance on the 9 datasets included in the code repo:

**Table 1** (predict horizon: 3 steps)
| Dataset | MAE  | RMSE | MAPE(%) |
| -----   | ---- | ---- | ---- |
| METR-LA | 2.56 | 5.06 | 6.46 |
| PEMS-BAY | 1.23 | 2.48 | 2.63 |
| PEMS07 | 2.14 | 4.01 | 5.01 |
| Solar | 1.52 | 1.53 | 1.42 |
| Electricity | 0.04 | 0.06 | 14.77 |
| ECG | 0.05 | 0.07 | 10.58 |

**Table 2** (predict horizon: 12 steps)
| Dataset | MAE  | RMSE | MAPE |
| -----   | ---- | ---- | ---- |
| PEMS03 | 14.32 | 21.64 | 16.24 |
| PEMS04 | 20.24 | 32.15 | 10.03 |
| PEMS08 | 15.83 | 24.93 | 9.26 |
