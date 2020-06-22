import pandas as pd
import numpy as np
import testing

def prepare_data(location_training_data, location_testing_data):
    # Define the headings to be used
    headings = ['dependent', 'handicapped-infants', 'water-project-cost-sharing', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satillite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-use', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']

    # Read in the official training and testing data separately
    df = pd.read_csv(location_training_data, names=headings)

    # Convert every '?' to NaN, so Panda's built-in functions can be used
    df = df.where(df != '?')

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

    # Reset the indices
    df = df.reset_index(drop=True)

    df = df.replace('y', 1)
    df = df.replace('n', 0)

    df.loc[df['dependent'] == 'republican', 'dependent'] = -1
    df.loc[df['dependent'] == 'democrat', 'dependent'] = 1

    # normalize the rows with the L2 norm
    for index, row in df.iterrows():
        row /= np.linalg.norm(row.values)

    return df

# Load and prepare the training and testing data
all_data = prepare_data(location_training_data='C:/Users/jordi/PycharmProjects/svm/datasets/Voting Records/house-votes-84.data',
                        location_testing_data=None)

# Define the location where the buckets are stored
buckets_loc = 'C:/Users/jordi/PycharmProjects/svm/datasets/Voting Records/buckets_congressional_voting.csv'

testing.run_epsilon(all_data, buckets_loc)