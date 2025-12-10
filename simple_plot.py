import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sample_data_1_month.csv")

plt.figure(figsize=(10,5))
plt.plot(df["close"], label="Close Price")
plt.title("Close Price Plot")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
