# This file extracts each skill bin from the alleps_nolast.tsv file and splits them into
# different files. Makes managing files and bins easier.

import csv

# change the file name to the file you want to overwrite and change row[32] == 'str' to the bin
with open('extreme_expert.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    data = [row for row in reader if row[32] == 'Extreme Expert' or row[32] == 'skill_bin']

with open('extreme_expert.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(data)
