import pandas as pd

# Read our csv into variable 'home data'
#home_data = pd.read_csv('train.csv')
#home_data.describe()

melb_data = pd.read_csv('melb_data.csv')

#drop empty columns
melb_data = melb_data.dropna(axis=0)
#this will print out (when ran in terminal) all of our column headers
melb_data.columns

# using dot notation to get a series (single column) of data.
# In this case we use it to isolate "Price" data
y = melb_data.Price

# select multiple features (columns) with brackets []
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melb_data[melb_features]

# terminal: show data stats - count, mean, std, min, max, etc
x.describe()
# terminal: show first 5 lines of data
#x.head()

# this is just here so I can place a stop point somewhere
quit()
