import pandas as pd
import numpy as np
import testing

def prepare_data(location_training_data, location_testing_data):
    # Define the headings to be used
    headings = ['dependent', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring number', 'ring-type', 'spore-print-color', 'population', 'habitat']

    # Read in the official training and testing data separately
    df = pd.read_csv(location_training_data, names=headings)

    # column stalk-root has 2480 missing values, drop column instead of dropping values to prevent loosing many instances
    df = df.drop(columns=['stalk-root'])

    df.loc[df['dependent'] == 'p', 'dependent'] = -1
    df.loc[df['dependent'] == 'e', 'dependent'] = 1
    df['dependent'] = df['dependent'].astype('int').apply(pd.to_numeric, downcast="unsigned")

    df = pd.get_dummies(df)

    # normalize the rows with the L2 norm
    for index, row in df.iterrows():
        row /= np.linalg.norm(row.values)

    return df

# Load and prepare the training and testing data
all_data = prepare_data(location_training_data='C:/Users/jordi/PycharmProjects/svm/datasets/Mushroom/agaricus-lepiota.data',
                        location_testing_data=None)

# Define the location where the buckets are stored
buckets_loc = 'C:/Users/jordi/PycharmProjects/svm/datasets/Mushroom/buckets_mushroom.csv'

testing.run_epsilon(all_data, buckets_loc)