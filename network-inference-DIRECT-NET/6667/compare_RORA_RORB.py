#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dir_prefix = "/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"

data_path = "data/adata_imputed_combined_v3.csv"
#%%
data = pd.read_csv(f"{dir_prefix}/{data_path}", index_col=0, header=0)
print(data.head())

#%%
# add best fit line
sns.regplot(data=data, x="Rora", y="Rorb", scatter=True,
            line_kws={"color": "red", "alpha": 0.5})

plt.show()
# %%
RORA_RORB_ave = data[["Rora", "Rorb"]].mean(axis=1)
sns.regplot(data=data, x=RORA_RORB_ave, y="Rora", scatter=True,
            line_kws={"color": "red", "alpha": 0.5})
plt.show()

# %%
sns.regplot(data=data, x=RORA_RORB_ave, y="Rorb", scatter=True,
            line_kws={"color": "red", "alpha": 0.5})
# %%
data["RORA_RORB"] = RORA_RORB_ave
data.to_csv(f"{dir_prefix}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv")
# %%
