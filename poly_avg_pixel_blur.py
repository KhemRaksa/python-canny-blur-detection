import pandas as pd
df = pd.read_csv('poly_sub_for_blur_eval.csv')
df2 = df["2"].mean()

min_pix = df["5"].min()
max_pix = df["5"].max()

print("Min", min_pix)
print("Max", max_pix)


print("Average Ratio:" ,df2)
print("Set Ratio: ", df2/(224*224))
