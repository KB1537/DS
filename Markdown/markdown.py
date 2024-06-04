import pandas as pd
 
file_name_csv = "./growth_practices.csv"
file_name_md = file_name_csv.replace("csv", "md")
 
df = pd.read_csv(file_name_csv,encoding='utf-8')
 
print(df.head(9))
with open(file_name_md, 'w') as md:
    df.to_markdown(buf=md, tablefmt="grid")