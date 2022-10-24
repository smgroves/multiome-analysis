# Run 2000

## Notes about the run
This run is based off of barcode 55476, but without a few of the sink nodes in the network.

55476 is having trouble running because there are 68 nodes, and you can only have 64 nodes to give a binary index to each vertex in the `graph-tool` network. 

For this barcode, I pulled the rules, binarized_data, and split data (train/test) directly from 55476. The network for this run is `DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs.csv` without a few of the sink nodes (that won't affect the network dynamics), which was saved as `DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_wo_sinks.csv`. Based on the info about the full network at `DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs.csv`, it has the following nodes:

- Number of nodes: 68
- Nodes:  ['AHR', 'ARID5B', 'ASCL1', 'BACH1', 'BACH2', 'BBX', 'CREB1', 'CUX1', 'CUX2', 'EGR1', 'EHF', 'EPAS1', 'ESR1', 'ETS1', 'FOS', 'FOSB', 'FOXO3', 'GLIS3', 'GRHL2', 'HIF1A', 'HSF2', 'JUN', 'JUNB', 'JUND', 'KMT2A', 'LCOR', 'LMX1B', 'MECOM', 'MEIS2', 'NFATC2', 'NFE2L2', 'NFIA', 'NFIB', 'NFIX', 'NFKB1', 'NPAS2', 'NR3C2', 'PBX1', 'PKNOX2', 'PPARG', 'PRDM16', 'PROX1', 'RARB', 'REST', 'RORA', 'RORB', 'RUNX1', 'SIX1', 'SIX4', 'SMAD3', 'SOX5', 'SOX6', 'SOX9', 'SP100', 'STAT1', 'STAT2', 'STAT4', 'TBX15', 'TCF12', 'TCF4', 'TCF7L1', 'TCF7L2', 'TEAD1', 'THRB', 'ZBTB18', 'ZBTB20', 'ZBTB7A', 'ZEB1']
- Sources:  ['ZBTB18', 'ZBTB7A', 'REST', 'TCF4', 'RORA', 'SOX9']
- Sinks:  ['BACH2', 'KMT2A', 'SOX5', 'HIF1A', 'STAT2', 'LCOR', 'BACH1', 'THRB', 'AHR', 'FOSB', 'GLIS3', 'TCF12', 'SOX6', 'MECOM', 'NPAS2', 'EPAS1', 'ARID5B', 'ZBTB20', 'NFE2L2', 'TCF7L1', 'STAT4', 'BBX']

To decide which of the sinks to remove, I picked the ones with the worst fit validation in 55476. 

**'BACH2', .84, RUNX1,TCF7L2,ZEB1,JUND,FOS,EGR1,PPARG,JUNB**
'KMT2A', .896
'SOX5', .913
'HIF1A', .94
'STAT2', .917
**'LCOR', .846**
'BACH1', .866
'THRB', .894
'AHR', .933
'FOSB', .933
'GLIS3', .903
'TCF12', .941
'SOX6', .913
'MECOM', .913, RUNX1
'NPAS2', .942
**'EPAS1', .865**
'ARID5B', .876, SP100
'ZBTB20', 1
'NFE2L2', .931
'TCF7L1', .941
**'STAT4', .751**
**'BBX'.83**

## Summary
- Network: `DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_wo_sinks.csv`
- Rules: fit in 55476, with a few rules for sinks removed **(STAT4, BBX, BACH2, EPAS1, and LCOR)**
- data_split: split into training and testing (for rule fitting) in 55476 and copied into this folder as is
- binarized_data: binarized the split data from 55476 again in this folder (to match fewer nodes)
- Validation: done in 55476, but not redone here