from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset = load_diabetes()         ## Set Dataset to loaded set // includes all information
    print (dataset)
    print (dataset.data)              ## Extract data
    print(dataset.target)             ## Target independently
    print(dataset['feature_names'])   ## Extract column

    df = pd.DataFrame(data=np.c_[dataset['data'], dataset['target']], columns = dataset['feature_names'] + ['target'])
    print(df)

    print (df.isnull().any())         ## Checks for any empty values

    for column in df:
        plt.figure()
        df.boxplot([column])
    plt.show()