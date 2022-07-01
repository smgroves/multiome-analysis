import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../data/arc_distance_inputed-ne.csv', header = 0, index_col=0)
[num_cells, num_arc] = data.shape
fraction = 10


print("Number of cells: ", num_cells)
print("Percentage close cells chosen: ", fraction)
divider = 100/fraction
for arc in data.drop('sample_id', axis = 1).columns:
    closest = (data.sort_values(arc, ascending = True).iloc[range(int(num_cells/divider))])
    data[f"{arc}_closest"] = [{True:1, False:0}[i] for i in (data.sample_id.isin(closest.sample_id))]
    data = data.drop(arc, axis = 1)
df = pd.melt(data, id_vars = ['sample_id'], value_vars = ['archetype_1_closest','archetype_2_closest','archetype_3_closest','archetype_4_closest'])
df = df.loc[df.value == 1]
print('Cells close to two archetypes: ',(df.sample_id.value_counts()==2).sum())
df = df.drop_duplicates(subset = 'sample_id', keep = "first")

df2 = pd.merge(left = data,right = df, how = 'left')
df2 = df2.fillna('None')
print(df2.head())
# df2[["sample_id","variable"]].to_csv('../data/labels_arc_dist.csv', index=False)
# print(df2.variable.value_counts())
df3 = pd.read_csv('../data/NE_cell_clusters.txt', header = 0, index_col=None, sep= '\t')
df3['sample_id'] = df3.index
print("Shape of clusters df: ", df3.shape)

df4 = pd.merge(left = df2[['sample_id','variable']], left_on='sample_id', right = df3, right_on= 'sample_id')
print("Shape after merge: ", df4.shape)
sns.heatmap(pd.crosstab(df4.variable, df4.x))
# plt.show()

df4.to_csv('../data/arc_distance_vs_cluster.csv')



ct = pd.crosstab(df4.x, df4.variable)
print(ct)
ct.columns = ['Generalist','1','2','3','4']
ct['variable'] = ct.index

plt.figure()
ct.plot.bar(stacked = True,
            color = ['Grey','Blue','Orange','Green','Red'])
plt.xticks(rotation = 45)
plt.title("Subtype Fractions by NE cluster")
# plt.show()
plt.savefig('./plots/Subtype-Fractions-by-NE-cluster.pdf')

ct.to_csv('../data/arc_distance_vs_cluster_sums.csv')
ct = ct.drop('Generalist', axis=  1)
plt.figure()
ct.plot.bar(stacked = True,
            color = ['Blue','Orange','Green','Red'])
plt.xticks(rotation = 45)
plt.title("Archetype Specialist Fractions by NE cluster")
# plt.show()
plt.savefig('./plots/Archetype-Specialist-Fractions-by-NE-cluster.pdf')


df4_spec = df4.loc[df4.variable != "None"]
ct_norm = pd.crosstab(df4_spec.x, df4_spec.variable, normalize='index')
ct_norm.columns = ['1','2','3','4']
ct_norm['variable'] = ct_norm.index

plt.figure()
ct_norm.plot.bar(stacked = True,
                 color = ['Blue','Orange','Green','Red'])
plt.xticks(rotation = 45)
plt.title("Normalized Archetype Specialist Fractions by NE cluster")

# plt.show()
plt.savefig('./plots/Archetype-Specialist-Fractions-by-NE-cluster-normalized.pdf')

#
# plt.figure()
# sns.barplot(data = ct, x = "variable", y = 'NE_1')
# plt.show()