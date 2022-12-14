

# Download and import the following dataset using pandas:
#   url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
#   column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
#   raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

import pandas as pd

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

# Print the mean values of each column.
print(raw_dataset.mean())

# Filter results by selecting only entries where the number of cylinders is equal to 3.
#print(raw_dataset[raw_dataset['Cylinders']==3])

# Filter results by selecting only entries where the number of cylinders is equal to 3 or 4.
print(raw_dataset[raw_dataset['Cylinders'].isin([3,4])])






