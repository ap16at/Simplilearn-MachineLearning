import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    north_america = pd.read_csv('north_america_2000_2010.csv',index_col = 0)
    south_america = pd.read_csv('south_america_2000_2010.csv',index_col = 0)

    americas = pd.concat([north_america,south_america])

    print("\nNorth America: ")
    print(north_america)
    print("\nSouth America: ")
    print(south_america)
    print("\nAmericas: ")
    print(americas)