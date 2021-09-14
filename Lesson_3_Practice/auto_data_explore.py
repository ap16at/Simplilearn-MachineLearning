import numpy, pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


def correlations_plot(dataframe):
    df = dataframe
    correlations = df.corr()
    sns.heatmap(data = correlations, square = True, cmap = "bwr")
    plt.yticks(rotation = 0)
    plt.xticks(rotation = 90)
    plt.show()


def missing_value_detection(dataframe):
    df = dataframe
    print("Checking for Missing Data:\n")
    print (df.isnull().any())


def mean_imputation(dataframe, cols):
    df = dataframe
    missing_col = cols
    for i in missing_col:
        df.loc[df.loc[:,i].isnull(),i]=df.loc[:,i].mean()
    print (df)


def median_imputation(dataframe, cols):
    df = dataframe
    missing_col = cols
    for i in missing_col:
        df.loc[df.loc[:,i].isnull(),i]=df.loc[:,i].median()
    print (df)


def outlier_detection(dataframe, col):
    df = dataframe
    sns.boxplot(x = df[col])
    plt.show()


def outlier_treatment(dataframe, col):
    df = dataframe
    filter = df[col].values > 300
    df_outlier_rem = df[filter]
    print(df_outlier_rem)


if __name__ == "__main__":
    # Import the dataset, explore for dimensionality, 
    # and type and average value of the horsepower across all the cars. 
    # Also, identify a few of mostly correlated features, 
    # which would help in modification.

    df = pandas.read_csv('mtcars.csv',delimiter = ',')
    model = numpy.array(df['model'])
    mpg = numpy.array(df['mpg'])
    cyl = numpy.array(df['cyl'])
    disp = numpy.array(df['disp'])
    hp = numpy.array(df['hp'])
    drat = numpy.array(df['drat'])
    wt = numpy.array(df['wt'])
    qsec = numpy.array(df['qsec'])
    vs = numpy.array(df['vs'])
    am = numpy.array(df['am'])
    gear = numpy.array(df['gear'])
    carb = numpy.array(df['carb'])

    avg_hp = numpy.mean(hp)

    print("Dataframe shape: " + str(df.shape))
    print("Horsepower datatype: " + str(df['hp'].dtype))
    print("Average Horsepower: " + str(avg_hp))
    correlations_plot(df)

    # Check for missing values and outliers within the horsepower column and remove them
    missing_value_detection(df)
    if df.isnull().values.any():
        column = input("Type column with missing values: ")
        decision = input("Impute with \"mean\" or \"median\"? : ")

        if decision == "mean":
            mean_imputation(df, [column])
        elif decision == "median":
            median_imputation(df, [column])

    outlier_detection(df, 'hp')
    print("Outlier lies above 300 horsepower.")
    outlier_treatment(df, 'hp')