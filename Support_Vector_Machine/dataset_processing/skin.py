import pandas as pd
import numpy as np
import testing

def prepare_data(location_training_data, location_testing_data):
    # Define the headings to be used
    headings = ['B', 'G', 'R', 'dependent']

    # Read in the official training and testing data separately
    df = pd.read_csv(location_training_data, names=headings)

    # Apply some pre-processing
    df.loc[df['dependent'] == 2, 'dependent'] = -1

    # Normalize the columns to contain values of max 1
    df /= df.max()

    for index, row in df.iterrows():
        row /= np.linalg.norm(row.values)

    print(df)

    return df

# Load and prepare the training and testing data
all_data = prepare_data(location_training_data='C:/Users/jordi/PycharmProjects/svm/datasets/Skin/skindata.csv',
                        location_testing_data=None)

# Define the location where the buckets are stored
buckets_loc = 'C:/Users/jordi/PycharmProjects/svm/datasets/Skin/buckets_skin.csv'

testing.run_epsilon(all_data, buckets_loc)