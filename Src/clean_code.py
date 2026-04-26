import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\UB Technology\PY-P\Customer Churn Prediction\Data\Churn_Modelling.csv")
# print(df)

# print(df.describe())
# print(df.info())

# for col in df.columns :
#        if pd.api.types.is_integer_dtype(df[col]) :
#             pass
#        else :
#              print(col)


# print(df.isna().sum())

#  check a output a categories distribution

# values = df.Exited.value_counts()
# labels = ['Not Exited', 'Exited']

# fig, ax = plt.subplots(figsize = (4, 3), dpi = 100)
# explode = (0, 0.09)

# patches, texts, autotexts = ax.pie(values, labels = labels, autopct = '%1.2f%%', shadow = True,
#                                    startangle = 90, explode = explode)

# plt.setp(texts, color = 'grey')
# plt.setp(autotexts, size = 8, color = 'white')
# autotexts[1].set_color('black')
# plt.show()

df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], axis=1)
# print(df.head())
# print(df.Geography.value_counts())

print(df.dtypes)
print(df.head())
import os 

df.to_csv("../Data/cleaned.csv", index=False)
