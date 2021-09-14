import pandas as pd

if __name__ == "__main__":
    dataset = pd.DataFrame({'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                        'last_name': ['Miller', 'Jacobson', '.', 'Milner','Cooze'],
                        'age': [42, 52, 36, 24, 73],
                        'preTestScore': [4, 24, 31, '.', '.'],
                        'postTestScore''': ['25,000', '94,000', 57, 62, 70]})
    dataset.to_csv('project.csv', index=False)
    df = pd.read_csv('project.csv')
    print(df)
    df = pd.read_csv('project.csv', skiprows=1, header=None)
    print(df)
    df = pd.read_csv('project.csv', index_col=['First Name', 'Last Name'], names=['UID', 'First Name', 'Last Name', 'Age', 'Pre-Test Score', 'Post-Test Score'])
    print(df)
    df = pd.read_csv('project.csv', na_values=['.'])
    print(pd.isnull(df))
    df = pd.read_csv('project.csv', skiprows=3)
    print(df)