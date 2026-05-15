# %%
import pandas as pd
import numpy as np
tmp = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["X", 'Y', 'Z'])
print(tmp)
# %%
sources = ["Z"]
small = tmp.loc[:, ~tmp.columns.isin(sources)]
print(small)
# %%
print(~tmp.columns.isin(sources))
# %%
l = [2, 4, 6, 8, 10]
l = [i for x, i in enumerate(l) if ~tmp.columns.isin(sources)[x]]
# %%
