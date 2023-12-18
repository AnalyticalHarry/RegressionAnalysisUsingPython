before_rows = df.shape[0]

def removing_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

for i in df.columns:
    df = removing_outliers_iqr(df, i)

filtered = df.shape[0]
print(f"Before shape was {before_rows} and after removing outliers our shape is {filtered}")