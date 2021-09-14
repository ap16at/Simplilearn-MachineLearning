import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('Salaries.csv', low_memory=False)

    ## How much total salary cost has increased from year 2011 to 2014?
    grouped_df = df.groupby(['Year']).sum().filter(['TotalPay'])
    total_pay_2011 = grouped_df.query('Year == 2011')['TotalPay']
    total_pay_2014 = grouped_df.query('Year == 2014')['TotalPay']
    
    total_change = float(total_pay_2014) - float(total_pay_2011)
    print("The total salary cost has increased ", str(total_change), " from 2011 to 2014.")

    ## Who was the top-earning employee across all the years?
    max_pay = df['TotalPay'].max()
    top_earner = df.query('TotalPay == ' + str(max_pay))['EmployeeName']

    print("The top earner across all the years is: ", str(top_earner))