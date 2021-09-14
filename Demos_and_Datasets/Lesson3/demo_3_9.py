import pandas as pd


if __name__ == "__main__":
    df = pd.readf = pd.read_csv('middle_tn_schools.csv')
    print(df.describe())
    print(df[['reduced_lunch','school_rating']].groupby(['school_rating']).describe())
    print(df[['reduced_lunch','school_rating']].corr())