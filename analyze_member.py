import pandas as pd
import sys

with open('member_info.txt', 'w', encoding='utf-8') as f:
    df = pd.read_csv('d:/Taiwei/RestaurantBookings/data/raw/member.csv', sep='\t', encoding='utf-16', on_bad_lines='warn')
    
    f.write("--- INFO ---\n")
    df.info(buf=f)
    
    f.write("\n--- NULL COUNT ---\n")
    f.write(df.isnull().sum().to_string() + "\n")
    
    f.write("\n--- HEAD ---\n")
    f.write(df.head().to_string() + "\n")
    
    f.write("\n--- UNIQUE COUNTS ---\n")
    f.write(df.nunique().to_string() + "\n")
    
    f.write("\n--- VALUE COUNTS FOR CATEGORICAL ---\n")
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'id':  # Skip id column to avoid huge output
            f.write(f"\n[{col}]\n")
            f.write(df[col].value_counts(dropna=False).head(20).to_string() + "\n")
