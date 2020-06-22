import pandas as pd
import numpy as np
import testing

def prepare_data(location_training_data, location_testing_data):
    # Define the headings to be used
    headings = ['dependent', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22']

    # Read in the official training and testing data separately
    data_train = pd.read_csv(location_training_data, names=headings)
    data_test = pd.read_csv(location_testing_data, names=headings, skiprows=1)
    df = data_train.append(data_test)

    # divide into ['not_recom', 'recommend', 'very_recom'] and ['priority', 'spec_prior']
    df.loc[df['dependent'] == 0, 'dependent'] = -1

    # Convert every '?' to NaN, so Panda's built-in functions can be used
    df = df.where(df != ' ?')

    # Try to fill missing values with the mean (this only works for numerical attributes)
    df = df.fillna(df.mean())

    # Get the number of records in the data set
    n_records_before = df.shape[0]

    # If values are still missing (they must be categorical attributes), drop the rows with missing data
    df = df.dropna()

    # Get the number of records in the data set
    n_records_after = df.shape[0]

    # Print how many records containing NaN values got dropped
    n_records_dropped = n_records_before - n_records_after
    print(n_records_dropped, "records were dropped due to missing values.")
    print("This is", round(n_records_dropped / n_records_before * 100, 1), "% of the entire data set.")
    print("The resulting data set contains", n_records_after, "records.")

    # Normalize the columns to contain values of max 1
    df /= df.max()

    for index, row in df.iterrows():
        row /= np.linalg.norm(row.values)

    return df

# Load and prepare the training and testing data
all_data = prepare_data(location_training_data='C:/Users/jordi/PycharmProjects/svm/datasets/Heart/SPECT.train',
                        location_testing_data='C:/Users/jordi/PycharmProjects/svm/datasets/Heart/SPECT.test')

# Define the location where the buckets are stored
buckets_loc = 'C:/Users/jordi/PycharmProjects/svm/datasets/Heart/buckets_SPEC.csv'

testing.run_epsilon(all_data, buckets_loc)