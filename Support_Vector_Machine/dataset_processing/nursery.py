import pandas as pd
import numpy as np
import testing

def prepare_data(location_training_data, location_testing_data):
    # Define the headings to be used
    headings = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'dependent']

    # Read in the official training and testing data separately
    df = pd.read_csv(location_training_data, names=headings)

    # divide into ['not_recom', 'recommend', 'very_recom'] and ['priority', 'spec_prior']
    df.loc[df['dependent'] == 'not_recom', 'dependent'] = -1
    df.loc[df['dependent'] == 'recommend', 'dependent'] = -1
    df.loc[df['dependent'] == 'very_recom', 'dependent'] = -1
    df.loc[df['dependent'] == 'priority', 'dependent'] = 1
    df.loc[df['dependent'] == 'spec_prior', 'dependent'] = 1
    df['dependent'] = df['dependent'].astype('int').apply(pd.to_numeric, downcast="unsigned")

    # Replace Categorical variables with binary
    df = pd.get_dummies(df)

    # Normalize the columns to contain values of max 1
    df /= df.max()

    for index, row in df.iterrows():
        row /= np.linalg.norm(row.values)

    return df

# Load and prepare the training and testing data
all_data = prepare_data(location_training_data='C:/Users/jordi/PycharmProjects/svm/datasets/Nursery/nursery.data',
                        location_testing_data=None)

# Define the location where the buckets are stored
buckets_loc = 'C:/Users/jordi/PycharmProjects/svm/datasets/Nursery/buckets_nursery.csv'

testing.run_epsilon(all_data, buckets_loc)