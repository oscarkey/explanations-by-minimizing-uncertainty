This directory contains demo tabular datasets which are loaded by uces.tabular.TabularDataset.
There are two types of file:
    - .csv files which contain the data
    - .txt files which contain metadata which is used to generate CEs

You can add your own tabular dataset here and start generating CEs.

The data CSV files have the column name as the first row, and the data as the subsequent rows.
The metadata files contain a key-value pair on each line. There are 4 types of line:
    - ignore_columns: names of columns to ignore, e.g. an id
    - range: minimum and maximum value of each column, to bound the generated CEs
    - perturbation: the unit size of the changes made to a particular column when generating CEs
    - output_columns: names of the columns which are the targets of the classification task