import pandas as pd
import sys

fname = sys.argv[1]
lang1 = sys.argv[2]
lang2 = sys.argv[3]

col_names = ["en","fr","ko","ja"]
df = pd.read_csv(fname)
df.columns = col_names

filtered = df[[lang1, lang2]]

filtered.to_csv(f"wikifiltered_{lang1}_{lang2}_test.txt", sep=" ", index=False, header=False)

