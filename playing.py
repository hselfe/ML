import pandas as pd
df = pd.DataFrame({"Item": ["A", "B", "C", "D", "E"], "Rating": [1, 0, 0, 1, 0] })
print(df.apply(lambda column: (column==0).sum())[1])
