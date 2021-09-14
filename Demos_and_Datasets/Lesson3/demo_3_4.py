import numpy, pandas


if __name__ == "__main__":
    df = pandas.read_csv('SalaryGender.csv',delimiter = ',')
    salary = numpy.array(df['Salary'])
    gender = numpy.array(df['Gender'])
    phd = numpy.array(df['PhD'])
    age = numpy.array(df['Age'])
    print(df)