import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
brcd = '1112'
perturbations = f"{dir_prefix}/{brcd}/perturbations/clustered_perturb_plots/"
stats = pd.read_csv(f"{perturbations}/perturbation_stats.csv", header = 0, index_col = None )

nonNE = ["Arc_2",'Arc_4','Generalist_nonNE']

NE = []
for i,r in stats.iterrows():
    if r['cluster'] in (nonNE):
        NE.append("nonNE")
    else:
        NE.append("NE")

stats['NE'] = NE
print(stats.head())

summ = stats.groupby(["gene",'NE', 'perturb']).mean().reset_index()

loc = (summ.pivot(index = ['gene','perturb'], columns = 'NE',values = 'mean').reset_index())

plt.figure(figsize = (15,15),dpi = 300)
sns.scatterplot(data = loc, x = 'NE',y = 'nonNE', hue = 'perturb')
for i,r in loc.iterrows():
    plt.annotate(r['gene'], (r['NE'],r['nonNE']))
plt.axhline(0, linestyle = "--", color = 'gray')
plt.axvline(0, linestyle = "--", color = 'gray')
plt.xlabel("NE destabilization score (average across attractors)")
plt.ylabel("Non-NE destabilization score (average across attractors)")
plt.xlim(-.65, .2)
plt.ylim(-.65,.2)
plt.savefig(f"{perturbations}/NE_vs_nonNE_scatterplot.pdf")