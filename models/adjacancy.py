import numpy as np
import pandas as pd
from ce290_data.data_cleaning import data_processing


df = data_processing('ce290_data/data/mar2020.csv')
df.data_cleaning()
df = df.df

df2 = df.groupby(['departure','arrival']).sum('number_of_flights').reset_index()
df2.columns = ['departure', 'arrival', 'weight']

import networkx as nx

g = nx.from_pandas_edgelist(df2, source='departure', target='arrival', edge_attr='weight' )
a = nx.adjacency_matrix(g)
print(a.todense())