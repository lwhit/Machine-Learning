import pandas as pd
# Use sklearn to model our data
from sklearn.tree import DecisionTreeRegressor

# Read our csv into variable 'home data'
#home_data = pd.read_csv('train.csv')
#home_data.describe()

melb_data = pd.read_csv('melb_data.csv')

#drop rows with empty Price values
melb_data = melb_data.dropna(axis=0)
# this will print out all of our column headers
#print(melb_data.columns)

# using dot notation to get a series (single column) of data.
# In this case we use it to isolate "Price" data
y = melb_data.Price

# select multiple features (columns) with brackets []
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_data[melb_features]

# show data stats - count, mean, std, min, max, etc
print(X.describe(), '\n', '\n')
#  show first 5 lines of data
#print(X.head())

###################
# Time to model our data. With the sklearn library!
# Steps to building and using a model:
# -Define. What type of model will it be?
# -Fit. Capture patterns from provided data
# -Predict
# -Evaluate. How accurate is our model?

# Define model. Specify a number for random_state to ensure same results each run
melb_model = DecisionTreeRegressor(random_state=1)

# Fit model
melb_model.fit(X, y)

# in practice, you'll want to make predictions for new houses coming on the market
# rather than the houses we already have prices for. Let's /predict/

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melb_model.predict(X.head()))
print("Here's the price values for those 5 houses:")
print((melb_data.Price).head())

# All that ^ did was copy exact values from what we already had for X..
# fantastic prediction!   /s

########################
# Model Validation

# Checkpoint. Reminder:
# 'y' is the value we care about - what we want to predict
# 'X' is the training data, so to say. We use X to predict y

# Once we have our model (melb_model), we can calculate mean absolute error (MAE)
from sklearn.metrics import mean_absolute_error as MAE

predictedHomePrices = melb_model.predict(X)
# calling the MAE function returns the answer we need to print out
print("\nMean Abs Error:", MAE(y, predictedHomePrices))

# You need to validate predictions on data that WAS NOT used to
# form the prediction model. An easy way to do this is to split
# off some data before creating the prediction model and set it aside
# once you have a prediction model, validate it against this data
# that was set aside. This is called "validation data"

# The scikit-learn library can split data for us
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melb_model = DecisionTreeRegressor()
# Fit model
melb_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melb_model.predict(val_X)
print("MAE with train test split:", MAE(val_y, val_predictions))

# overfitting vs underfitting, both make your predictive model less
# accurate, i.e. increased MAE. Overfitting means your decision tree
# is too deep, i.e. too many tree nodes.
# Underfitting means decition tree too shallow; not enough nodes
# Our aim is to get decision tree depth just right. Here's a way
# we can test to see how deep our tree should be:

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = MAE(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# ^ that output shows that 500 is the best choice for this data,
# because it has the lowest MAE.

# Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions
# Underfitting: failing to capture relevant patterns, again leading to less accurate predictions

##########################
# We were able to tune our Decision Tree model, but it's not very
# sophisticated/accurate. On to the next strategy:

#### Random Forests ####
# random forest uses many decision trees, and it makes a prediction by averaging the
# predictions of each component tree. Generally better predictions that a single tree

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print("Random forest MAE: ", MAE(val_y, melb_preds))

