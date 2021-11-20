# Clean Read Counts from RNA-seq files
# Input: RNA seq read counts
# Output: Normalized FPM data



# From WGCNA FAQs:
# We suggest removing features whose counts are consistently low
# (for example, removing all features that have a count of less than say 10 in more than 90% of the samples)
# because such low-expressed features tend to reflect noise and correlations based on counts that are
# mostly zero aren't really meaningful. The actual thresholds should be based on experimental design,
# sequencing depth and sample counts.

import pandas as pd
import os
import re
import numpy as np
# Input read counts and save file for lung counts

directory = "/Users/smgroves/Dropbox (VU Basic Sciences)/SCLC_data/RNAseq/CCLE/"
# protein_coding = pd.read_csv(os.path.join(directory,"protein_coding.csv"))

#print(protein_coding.head())
if False:
    read_counts = pd.read_table(os.path.join(directory,"CCLE_DepMap_18Q1_RNAseq_reads_20180214.gct"), sep="\t")
    #read_counts = pd.read_table(os.path.join(directory, "CCLE_copyno_bygene.txt"), sep="\t")
    read_counts=read_counts.set_index(['Description'])
    #read_counts=read_counts.set_index(['SYMBOL'])

    del read_counts['Name']
    print(list(read_counts))
    # cell_lines=['NCIH1048','NCIH1341','SBC5','DMS114','NCIH841','SW1271','NCIH2286','NCIH1339','NCIH196','NCIH2066',
    #             'NCIH211','DMS273','CORL311','NCIH526','NCIH446','NCIH82','NCIH1694','NCIH2171','CPCN','NCIH524','CORL24',
    #             'HCC33','NCIH2227','SCLC21H','CORL88','CORL95','DMS153','DMS53','SHP77','DMS454','NCIH510','NCIH2196',
    #             'COLO668','NCIH2029','NCIH1836','CORL279','CORL51','NCIH1092','NCIH69','CORL47','NCIH2081','NCIH146',
    #             'NCIH889','NCIH1184','NCIH1930','NCIH1436','NCIH209','NCIH1876','NCIH1963','NCIH1105','NCIH2141','DMS79','NCIH1618']
    cell_lines = ['SKIN']
    #Get only small cell lung cancer columns

    for i in list(read_counts):
        true_count = 0
        for j in range(len(cell_lines)):
            if re.search(cell_lines[j],i):
                true_count = 1
                print(i)
                break # if it matches an SCLC cell line, break out of for loop and go to next column
        if true_count == 0:
            read_counts = read_counts.drop(i, axis=1)

    read_counts.to_csv(os.path.join(directory,"full_skin_reads.csv"))


# Input RPKM and save file for lung RPKM
if False:
    rpkm = pd.read_table(os.path.join(directory,"CCLE_DepMap_18Q1_RNAseq_RPKM_20180214.gct"), sep="\t")
    rpkm=rpkm.set_index(['Description'])
    print(rpkm.shape)
    del rpkm['Name']

    print(list(rpkm))
    # cell_lines=['NCIH1048','NCIH1341','SBC5','DMS114','NCIH841','SW1271','NCIH2286','NCIH1339','NCIH196','NCIH2066',
    #             'NCIH211','DMS273','CORL311','NCIH526','NCIH446','NCIH82','NCIH1694','NCIH2171','CPCN','NCIH524','CORL24',
    #             'HCC33','NCIH2227','SCLC21H','CORL88','CORL95','DMS153','DMS53','SHP77','DMS454','NCIH510','NCIH2196',
    #             'COLO668','NCIH2029','NCIH1836','CORL279','CORL51','NCIH1092','NCIH69','CORL47','NCIH2081','NCIH146',
    #             'NCIH889','NCIH1184','NCIH1930','NCIH1436','NCIH209','NCIH1876','NCIH1963','NCIH1105','NCIH2141','DMS79','NCIH1618']
# Get only small cell lung cancer columns
    cell_lines = ['SKIN']

    for i in list(rpkm):
        true_count = 0
        for j in range(len(cell_lines)):
            if re.search(cell_lines[j],i):
                true_count = 1
                print(i)
                break # if it matches an SCLC cell line, break out of for loop and go to next column
        if true_count == 0:
            rpkm = rpkm.drop(i, axis=1)
    print(rpkm.shape)
    rpkm.to_csv(os.path.join(directory,"full_skin_rpkm.csv"))


# Remove features with consistently low counts
# Alternatively: remove rows where 100% are <10,
# Figure out how many the below line removes:
# where 90% are <10 and the remaining 10% are <100 (or another good number)
def collapse_rows(df):
    gb = df.groupby(df.index)
    return gb.apply(np.mean)


if False:
    #lung_rpkm = pd.read_csv("/Users/sarahmaddox/Quaranta_Lab/SCLC_data/RomanThomasSCLC/expression_all.csv", index_col=0)
    lung_counts = pd.read_csv(os.path.join(directory,"full_skin_reads.csv"), index_col=[0])
    lung_rpkm = pd.read_csv(os.path.join(directory,"full_skin_rpkm.csv"), index_col=[0])
    #print(lung_counts.shape)
    #print(lung_counts.head())
    # lung_counts  = collapse_rows(lung_counts)
    lung_rpkm = collapse_rows(lung_rpkm)
    lung_rpkm.to_csv(os.path.join(directory,"skin_expression_rpkm_collapsed.csv"))


if False:
    #For read counts of CCLE: remove less than 10
    #For rpkm of RT data: remove less than .8
    lung_counts = pd.read_csv(os.path.join(directory,"skin_expression_rpkm_collapsed.csv"), index_col=0)
    #lung_counts = pd.read_csv('/Users/sarahmaddox/Quaranta_Lab/SCLC_data/Sage_mouse_clines/22_mouse_clines.csv', index_col=0)
    no_low_counts_lung = pd.DataFrame()
    no_low_counts_rpkm = pd.DataFrame()
    old_counts = pd.DataFrame()
    old_rpkm = pd.DataFrame()
    i=0
    for index, row in lung_counts.iterrows():
        low_count=0
        remaining_low_count = 0
        for col in list(lung_counts):
            if float(lung_counts.loc[index][col]) < .8:
                low_count = low_count + 1
            elif int(lung_counts.loc[index][col]) < 100:
                remaining_low_count = remaining_low_count + 1
        low_percent = low_count/len(list(lung_counts))
        remaining_low_percent = remaining_low_count/len(list(lung_counts))

        #Skip row if low_percent = 1...
        if low_percent == 1:
            print(index)
        # ... or if low percent is > .9 and low percent + remaining low percent = 1
        elif low_percent >.9 and (low_percent+remaining_low_percent == 1):
            print(index)
        else:
            print(i)
            no_low_counts_lung = no_low_counts_lung.append(row)
            #no_low_counts_rpkm = no_low_counts_rpkm.append(lung_rpkm.loc[index])
        i = i + 1

        #Append to df if low percent is <.9... alternative to above; change old_counts to no_low_counts_lung if necessary
        if low_percent < 0.9:
            print(i)
            old_counts = no_low_counts_lung.append(row)
            #old_rpkm = no_low_counts_rpkm.append(lung_rpkm.loc[index])
        else:
            print(index)

    print(no_low_counts_lung.shape)
    print("Removing all rows with 90% <10 counts:", old_counts.shape)

    no_low_counts_lung.to_csv(os.path.join(directory, "skin_rpkm_cleaned.csv"))
    #no_low_counts_rpkm.to_csv(os.path.join(directory, "lung_rpkm_cleaned.csv"))

#Take only protein coding genes
if False:
    lung_protein = pd.DataFrame()
    lung_clean = pd.read_csv(os.path.join(directory,"lung_rpkm_cleaned.csv"))
    #lung_clean = pd.read_csv("/Users/sarahmaddox/Quaranta_Lab/SCLC_data/CCLE/RNAseq_02_2018/lung_rpkm_cleaned.csv")
    for index, row in lung_clean.iterrows():
        for i in range(len(protein_coding)):
            if lung_clean.loc[index, 'Unnamed: 0'] == protein_coding.loc[i, 'Gene name']:
                lung_protein = lung_protein.append(row)
                print(index)
    print(lung_protein.head())
    lung_protein.to_csv(os.path.join(directory,"lung_protein_coding_cleaned.csv"))

############HARDCODED:################
#  Remove column Unnamed: 0 and replace first row (index numbers) with Unnamed: 0. Change name of column to "Gene"
######################################

# Collapse rows by average with same gene name with a period

if True:
    lung_rpkm_final = pd.read_csv(os.path.join(directory, "skin_rpkm_cleaned.csv"))
    #lung_rpkm_final = pd.read_csv(os.path.join(directory,"lung_protein_coding_cleaned.csv"))
    print(lung_rpkm_final.shape)

    for index, row in lung_rpkm_final.iterrows():

        # if re.match(r"\'(.*)\b(\d{1}|\d{2})\'", lung_rpkm_final.loc[index, 'Unnamed: 0']):
        #     match2 = re.match(r"\'(.*)\b(\d{1}|\d{2})\'", lung_rpkm_final.loc[index, 'Unnamed: 0'])
        #     lung_rpkm_final.loc[index, 'Unnamed: 0'] = match2.group(1)

        #split by a period and take the first part; if no period, it takes the whole string
        lung_rpkm_final.loc[index,'Gene'] = str(lung_rpkm_final.loc[index, 'Gene']).split('.')[0]

    lung_rpkm_collapse = lung_rpkm_final.groupby('Gene').mean()
    print(len(lung_rpkm_collapse))
    print(lung_rpkm_collapse.head())
    lung_rpkm_collapse.to_csv(os.path.join(directory, "skin_rpkm_collapsed.csv"))

    # FPM
    tpm_df = lung_rpkm_collapse/lung_rpkm_collapse.sum()*1000000
    tpm_df.to_csv(os.path.join(directory, "skin_tpm_collapsed.csv"))


###TRY ARCSINH transformation instead of log1p TODO

if True:
    #read in RPKM
    lung_norm= pd.read_csv(os.path.join(directory, "skin_tpm_collapsed.csv"), index_col=['Gene'])

    # Log-normalized: Set 0's to .1*min = David's way
    # Following Avi Ma'ayan's lab:
    # http://nbviewer.jupyter.org/github/maayanlab/Zika-RNAseq-Pipeline/blob/master/Zika.ipynb
    lung_norm = np.log1p(lung_norm)

    # for i, r in lung_norm.iterrows():
    #     for j in list(lung_norm):
    #         if lung_norm.loc[i,j] == 0:
    #             lung_norm.loc[i,j] == .000001
    # lung_norm = np.log10(lung_norm)

    # # Normalize between 0 and 1
    # #Don't do this yet, according to David. Can be normalize in analysis code later
    # lung_norm = lung_norm.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

    print(lung_norm.head())
    lung_norm.to_csv(os.path.join(directory, "skin_tpm_final.csv"))

# Hard coded: manually removed column "H2110," which is not SCLC but matched pattern "H211" in lung_tpm_final