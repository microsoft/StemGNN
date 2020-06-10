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
[ECG](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)



We can get the raw data through the link above. We evaluate the performance of traffic flow forecasting on PEMS03, PEMS07, PEMS08 and traffic speed forecasting on PEMS04, PEMS-BAY and METR-LA. So we use the traffic flow table of PEMS03, PEMS07, PEMS08 and the traffic speed table of PEMS04, PEMS-BAY and METR-LA as our datasets. We download the solar power data of Alabama (Eastern States) and merge the *5_Min.csv (totally 137 time series) as our Solar dataset. We delete the header and index of Electricity file downloaded from the link above as our Electricity dataset. During data processing, for the missing values ​​(0 values) in the data, we use the average value of the its time series (mean of the column) instead, which can avoid 0 values ​​when calculating MAPE. We name each file after the datasets. The *.csv is in shape of `N*T`, where `N` denotes total number of timestamps, `T` denotes number of nodes.

We provide a cleaned version of ECG ([./dataset/ECG_data.csv](./dataset/ECG_data.csv)) for reproduction convenience. The ECG_data.csv is in shape of `140*5000`, where `140` denotes total number of timestamps, `5000` denotes number of nodes. Run command `python main.py` to trigger training and evaluation on ECG_data.csv.

## Training and Evaluation

The training procedure and evaluation procedure are all included in the `main.py`. To train and evaluate on some dataset, run the following command:

For ECG dataset:

```train & evaluate ECG
python main.py --train True --evaluate True --dataset ./dataset/ECG_data.csv --output_dir ./output/ECG_data --n_route 140 --n_his 12 --n_pred 3
```

We set the flag 'train' to 'True' so that we can train our model and set the flag 'evaluate' to 'True' so that we can evaluate our model after we save the model to the flag 'output_dir'. StemGNN reads data from 'dataset'. Besides, the flag 'n_route' means the number of time series, the 'n_his' is our sliding window and the 'n_pred' is the horizon.


### Complete settings for all datasets

**Table 1** (Settings for datasets)
| Dataset | train  | evaluate | dataset | output_dir | n_route | n_his | n_pred|
| -----   | ---- | ---- | ---- |---- |---- |---- |---- |
| PEMS03 | True | True | ./dataset/PEMS03 | ./output/PEMS03 | 358 | 12 | 12 | 
| PEMS04 | True | True | ./dataset/PEMS04 | ./output/PEMS04 | 307 | 12 | 12 | 
| PEMS08 | True | True | ./dataset/PEMS08 | ./output/PEMS08 | 170 | 12 | 12 |
| METR-LA | True | True | ./dataset/METR-LA | ./output/METR-LA | 207 | 12 | 3 |
| PEMS-BAY | True | True | ./dataset/PEMS-BAY | ./output/PEMS-BAY | 325 | 12 | 3 |
| PEMS07 | True | True | ./dataset/PEMS07 | ./output/PEMS07 | 228 | 12 | 3 |
| Solar | True | True | ./dataset/Solar | ./output/Solar | 137 | 12 | 3 |
| Electricity | True | True | ./dataset/Electricity | ./output/Electricity | 321 | 12 | 3 |






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
| ECG | 0.05 | 0.07 | 10.58 |

**Table 3** (predict horizon: 12 steps)
| Dataset | MAE  | RMSE | MAPE |
| -----   | ---- | ---- | ---- |
| PEMS03 | 14.32 | 21.64 | 16.24 |
| PEMS04 | 20.24 | 32.15 | 10.03 |
| PEMS08 | 15.83 | 24.93 | 9.26 |
