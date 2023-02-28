import pandas as pd

file = 'jan2020.csv'
df = pd.read_csv(f'ce290_data/{file}', header=1).iloc[:-5,:]
df = df[['Departure', 'Arrival', 'Date', 'Flight\nCount', 'Airport\nDeparture\nDelay', 'Airborne\nDelay','Gate\nArrival\nDelay']]
df.columns = ['depature', 'arrival', 'date', 'number_of_flights', 'departure_airport_delay', 'airborne_delay','arrival_gate_delay']
df.head(10)

