import pandas as pd

class data_processing:
    """
    Process the data from specified path.
    """
    def __init__(self, path):
        self.path = path

    def data_cleaning(self):
        # Use vscode
        df = pd.read_csv(f'./{self.path}').reset_index(drop=True).iloc[5:-5, [0,1,2,3,12,15,16]]
        
        # Use PyCharm
        # df = pd.read_csv(f'./{self.path}', header=1).iloc[:-5, :]
        # df = df[['Departure',
        #          'Arrival',
        #          'Date',
        #          'Flight\r\nCount',
        #          'Airport\r\nDeparture\r\nDelay',
        #          'Airborne\r\nDelay',
        #          'Gate\r\nArrival\r\nDelay']]
        df.columns = ['departure',
                      'arrival',
                      'date',
                      'number_of_flights',
                      'departure_airport_delay',
                      'airborne_delay',
                      'arrival_gate_delay']
        pd.set_option('display.max_columns', None)
        df = df.iloc[1:,:]
        df['number_of_flights'] = df['number_of_flights'].astype(int)
        self.df = df
    
    def parse_x_matrix(self):
        # Group by dep and arr airports
        dep = self.df.groupby(['departure', 'date']).sum('number_of_flights').reset_index()[['departure', 'number_of_flights', 'date']]
        arr = self.df.groupby(['arrival', 'date']).sum('number_of_flights').reset_index()[['arrival', 'number_of_flights', 'date']]
        # Merge dep and arr aggregated numbers
        merged = arr.merge(dep, left_on=['arrival','date'], right_on=['departure', 'date'])
        merged['total_flights'] = merged['number_of_flights_x'] + merged['number_of_flights_y']
        merged = merged[['departure', 'total_flights', 'date']]
        # Create the pivoted matrix
        pivoted = pd.pivot(merged, index='departure', columns='date', values='total_flights')
        pivoted.fillna(0, inplace=True)
        self.x_matrix = pivoted
        return self.x_matrix
    


## Example
# processor = data_processing('ce290_data/data/feb2020.csv')
# processor.data_cleaning()
# x_matrix = processor.parse_x_matrix()

