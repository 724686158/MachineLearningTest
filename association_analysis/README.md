Apriori Algorithm and FP-Growth Algorithm
==========================================
Implemented by python 3.7 

Without unofficial module requirements

code from
https://github.com/asaini/Apriori
https://github.com/evandempsey/fp-growth

List of files
-------------
0. association_analysis.py
1. apriori.py
2. fp_growth.py
3. tesco.csv
4. README(this file)
5. data_process_tools.py
6. input_tools.py
7. out_tools.py
8. INTEGRATED-DATASET.csv

The dataset is a copy of the “Online directory of certified businesses with a detailed profile” file from the Small Business Services (SBS) 
dataset in the `NYC Open Data Sets <http://nycopendata.socrata.com/>`_

Usage
-----
To run the program with dataset provided and default values for *minSupport* = 0.15 and *minConfidence* = 0.6

    python association_analysis.py -m apriori -f INTEGRATED-DATASET.csv

    python association_analysis.py -m fp-growth -f INTEGRATED-DATASET.csv


To run program with dataset  

     python association_analysis.py -m apriori -f INTEGRATED-DATASET.csv -s 0.17 -c 0.68

     python association_analysis.py -m fp-growth -f INTEGRATED-DATASET.csv -s 0.17 -c 0.68

data structure
-------
input file:  *.csv

output data: items, rules
     
     items : [(item, support)]
     
     roules : [(((element), (remain)), confidence)]

License
-------
MIT-License

