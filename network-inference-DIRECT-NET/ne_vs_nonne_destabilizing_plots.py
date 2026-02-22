#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import booleabayes as bb
import os
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
brcd = '1112'

print(brcd)
#%%
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
print(summ.head())

#%%
loc = (summ.pivot(index = ['gene','perturb'], columns = 'NE',values = 'mean').reset_index())

# plt.figure(figsize = (15,15),dpi = 300)
# sns.scatterplot(data = loc, x = 'NE',y = 'nonNE', hue = 'perturb')
# for i,r in loc.iterrows():
#     plt.annotate(r['gene'], (r['NE'],r['nonNE']))
# plt.axhline(0, linestyle = "--", color = 'gray')
# plt.axvline(0, linestyle = "--", color = 'gray')
# plt.xlabel("NE destabilization score (average across attractors)")
# plt.ylabel("Non-NE destabilization score (average across attractors)")
# plt.xlim(-.65, .2)
# plt.ylim(-.65,.2)
# # plt.savefig(f"{perturbations}/NE_vs_nonNE_scatterplot.pdf")
# plt.show()

perturb_folders =  f"{dir_prefix}/{brcd}/perturbations/"
ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"

attractor_dict = {}
attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
for i,r in attr_filtered.iterrows():
    attractor_dict[i] = []

for i,r in attr_filtered.iterrows():
    attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))

print(attractor_dict)

p = [ name for name in os.listdir(perturb_folders) if os.path.isdir(os.path.join(perturb_folders, name)) ]

genes = sorted(list(set(stats.gene)))
results_df = pd.DataFrame(columns=genes)

indices = []
act = []
for phen in attractor_dict:
    res_act_dict = {}
    res_kd_dict = {}
    for g in genes:
        res_act_dict[g] = 0
        res_kd_dict[g] = 0
    print(phen)
    for f in attractor_dict[phen]:
        if f == "clustered_perturb_plots": continue
        else:
            res = pd.read_csv(f"{perturb_folders}/{f}/results.csv", header = None, index_col = None)
            res_act = res.loc[res[3] == 'activate']
            res_kd = res.loc[res[3]=='knockdown']
            for g in genes:
                if res_act.loc[res_act[2]==g,4].mean() < -.03: res_act_dict[g] += 1
                if res_kd.loc[res_kd[2]==g,4].mean() < -0.3: res_kd_dict[g] += 1
    print(res_act_dict)
    print(res_kd_dict)
    results_df = results_df.append(res_act_dict, ignore_index=True)
    results_df = results_df.append(res_kd_dict, ignore_index=True)
    indices.append(f'{phen}_act')
    indices.append(f'{phen}_kd')
    act.append('act')
    act.append('kd')
results_df.index = indices
results_df['act'] = act

# results_df.to_csv(f"{perturbations}/significant_attractor_perturbations_-0.3.csv")

loc['num_attr'] = 0
act_sum = results_df.loc[results_df['act'] == 'act'].sum()
kd_sum = results_df.loc[results_df['act'] == 'kd'].sum()
for i,r in loc.iterrows():
    if r['perturb'] == 'activate':
        loc.loc[i,'num_attr'] = act_sum[r['gene']]
    elif r['perturb'] == 'knockdown':
        loc.loc[i,'num_attr'] = kd_sum[r['gene']]

loc['size'] = 4*loc['num_attr']

#%%
plt.figure(figsize = (15,15),dpi = 300)
sns.scatterplot(data = loc, x = 'NE',y = 'nonNE', hue = 'perturb', size='num_attr', sizes = (20,500))
for i,r in loc.iterrows():
    plt.annotate(r['gene'], (r['NE'],r['nonNE']))
plt.axhline(0, linestyle = "--", color = 'gray')
plt.axvline(0, linestyle = "--", color = 'gray')
plt.xlabel("NE destabilization score (average across attractors)")
plt.ylabel("Non-NE destabilization score (average across attractors)")
plt.xlim(-.65, .2)
plt.ylim(-.65,.2)
# plt.savefig(f"{perturbations}/NE_vs_nonNE_scatterplot_KD_only.pdf")
plt.show()

#%%
loc_KD = loc.loc[loc['perturb'] == 'activate']
plt.figure(figsize = (15,10),dpi = 300)
sns.scatterplot(data = loc_KD, x = 'NE',y = 'nonNE',  sizes = (20,500))
for i,r in loc_KD.iterrows():
    if r['NE'] < -.05 or r['nonNE'] < -0.05:
        plt.annotate(r['gene'], (r['NE'],r['nonNE']), ha = 'right')
plt.axhline(0, linestyle = "--", color = 'gray')
plt.axvline(0, linestyle = "--", color = 'gray')
# add highlight where x < r.loc['ASCL1','NE'] 
# ascl1_x = loc_KD.loc[loc_KD['gene'] == 'ASCL1', 'NE'].iloc[0]
# plt.axvline(ascl1_x, linestyle = "--", color = 'red')

plt.xlabel("NE destabilization score (average across attractors)")
plt.ylabel("Non-NE destabilization score (average across attractors)")
plt.xlim(-.4, .2)
plt.ylim(-.65,.2)
plt.savefig(f"{perturbations}/NE_vs_nonNE_scatterplot_act_only.pdf")
# plt.show()
#%% Plot only top master regulators for each archetype independently
# only use information from KD perturbations, since activation would suggest master destabilizer

single_stats = stats.loc[stats['perturb'] == 'knockdown']

# wide form not long form: rows = gene, column names = cluster, values = mean destabilization score
single_stats_wide = single_stats.pivot(index = 'gene', columns = 'cluster', values = 'mean').reset_index()

#rename particular columns
single_stats_wide = single_stats_wide.rename(columns = {'Arc_1':'Intermediate','Arc_3':'Secretory','Arc_4':'nonNE1','Arc_5':'NE1', 'Arc_6':'NE2'})
(single_stats_wide.head())
# %%
# make a new df that for each column calculates [column] - [ave of other columns] to get a relative destabilization score for each gene in each archetype
archetypes = ['Intermediate','Secretory','nonNE1','NE1','NE2','Arc_1_Generalist_NE','Arc_5_Generalist_NE','Generalist_NE','Generalist_nonNE']
relative_destabilization = pd.DataFrame(columns = ['gene'] + archetypes)
relative_destabilization['gene'] = single_stats_wide['gene']
for a in archetypes:
    relative_destabilization[a] = single_stats_wide[a] - single_stats_wide[[c for c in archetypes if c != a]].mean(axis = 1)        
# %%
