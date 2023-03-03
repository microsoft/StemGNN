import pandas as pd

class data_processing:
    """
    Process the data from specified path.
    """
    def __init__(self, path):
        self.path = path

    def data_cleaning(self):
        df = pd.read_csv(f'./{self.path}', header=1)
        # df = pd.read_csv(f'./{self.path}', header=1).iloc[:-5, :]
        df = df[['Departure',
                 'Arrival',
                 'Date',
                 'Flight\r\nCount',
                 'Airport\r\nDeparture\r\nDelay',
                 'Airborne\r\nDelay',
                 'Gate\r\nArrival\r\nDelay']]
        df.columns = ['depature',
                      'arrival',
                      'date',
                      'number_of_flights',
                      'departure_airport_delay',
                      'airborne_delay',
                      'arrival_gate_delay']
        pd.set_option('display.max_columns', None)
        # print(df.head(10))
        return df

## Example
# processor = data_processing('jan2020.csv')
# df_1 = processor.data_cleaning()
# print(df_1)

