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
[METR-LA](https://github.com/liyaguang/DCRNN),
[PEMS-BAY](https://github.com/liyaguang/DCRNN),
[Solar](https://www.nrel.gov/grid/solar-power-data.html),
[Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014),
[ECG5000](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)
[COVID-19](https://github.com/CSSEGISandData/COVID-19/tree/master)




We can get the raw data through the link above. We evaluate the performance of traffic flow forecasting on PEMS03, PEMS07, PEMS08 and traffic speed forecasting on PEMS04, PEMS-BAY and METR-LA. So we use the traffic flow table of PEMS03, PEMS07, PEMS08 and the traffic speed table of PEMS04, PEMS-BAY and METR-LA as our datasets. We download the solar power data of Alabama (Eastern States) and merge the *5_Min.csv (totally 137 time series) as our Solar dataset. We delete the header and index of Electricity file downloaded from the link above as our Electricity dataset. For COVID-19 dataset, the raw data is at the floder `csse_covid_19_data/csse_covid_19_time_series/` of the above github link. We use `time_series_covid19_confirmed_global.csv` to calculate the daily of newly confirmed people number from 1/22/2020 to 5/10/2020. The 25 countries we used are 'US','Canada','Mexico','Russia','UK','Italy','Germany','France','Belarus ','Brazil','Peru','Ecuador','Chile','India','Turkey','Saudi Arabia','Pakistan','Iran','Singapore','Qatar','Bangladesh','Arab','China','Japan','Korea'. During data processing, we use mean value to replace 0 values in sort of dataset (e.g. PEMS08, Electricity) to avoid overflow in calculating MAPE. We name each file after the datasets. The *.csv is in shape of `N*T`, where `N` denotes number of nodes, `T` denotes total number of timestamps.

We provide a cleaned version of ECG5000 ([./dataset/ECG_data.csv](./dataset/ECG_data.csv)) for reproduction convenience. The ECG_data.csv is in shape of `140*5000`, where `140` denotes total number of nodes, `5000` denotes number of timestamps. Run command `python main.py` to trigger training and evaluation on ECG_data.csv.

## Training and Evaluation

The training procedure and evaluation procedure are all included in the `main.py`. To train and evaluate on some dataset, run the following command:

For ECG5000 dataset:

```train & evaluate ECG
python main.py --train True --evaluate True --dataset ./dataset/ECG_data.csv --output_dir ./output/ECG_data --n_route 140 --n_his 12 --n_pred 3
```

We set the flag 'train' to 'True' so that we can train our model and set the flag 'evaluate' to 'True' so that we can evaluate our model after we save the model to the flag 'output_dir'. StemGNN reads data from 'dataset'. Besides, the flag 'n_route' means the number of time series, the 'n_his' is our sliding window and the 'n_pred' is the horizon.


### Complete settings for all datasets

**Table 1** (Settings for datasets)
| Dataset | train  | evaluate  | n_route | n_his | n_pred|
| -----   | ---- | ---- |---- |---- |---- |
| PEMS03 | True | True |  358 | 12 | 12 | 
| PEMS04 | True | True |  307 | 12 | 12 | 
| PEMS08 | True | True |  170 | 12 | 12 |
| METR-LA | True | True | 207 | 12 | 3 |
| PEMS-BAY | True | True |  325 | 12 | 3 |
| PEMS07 | True | True | 228 | 12 | 3 |
| Solar | True | True |  137 | 12 | 3 |
| Electricity | True | True | 321 | 12 | 3 |
| ECG5000| True | True | 140 | 12 | 3 |
| COVID-19| True | True | 25 | 28 | 28 |


In this code repo, we have processed ECG5000 as the sample dataset, the input is stored  at `./dataset/ECG_data.csv` and the output of StemGNN will be stored at `./output/ECG_data`.




## Results

Our model achieves the following performance on the 9 datasets included in the code repo:

**Table 2** (predict horizon: 3 steps)
| Dataset | MAE  | RMSE | MAPE(%) |
| -----   | ---- | ---- | ---- |
| METR-LA | 2.56 | 5.06 | 6.46 |
| PEMS-BAY | 1.23 | 2.48 | 2.63 |
| PEMS07 | 2.14 | 4.01 | 5.01 |
| Solar | 1.52 | 1.53 | 1.42 |
| Electricity | 0.04 | 0.06 | 14.77 |
| ECG5000 | 0.05 | 0.07 | 10.58 |

**Table 3** (predict horizon: 12 steps)
| Dataset | MAE  | RMSE | MAPE |
| -----   | ---- | ---- | ---- |
| PEMS03 | 14.32 | 21.64 | 16.24 |
| PEMS04 | 20.24 | 32.15 | 10.03 |
| PEMS08 | 15.83 | 24.93 | 9.26 |

**Table 4** (predict horizon: 28 steps)
| Dataset | MAE  | RMSE | MAPE |
| -----   | ---- | ---- | ---- |
| COVID-19 | 662.24 | 1023.19| 19.3|

