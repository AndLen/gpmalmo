# gptsne
Rough usage (from the src/ directory):   
`python3 -m gpmalmo.gpmal_nc --help`  
e.g. `python3 -m gpmalmo.gpmal_nc -d COIL20 --dir "datasets/"`

* Datasets used in the paper are in datasets/
* Add your own datasets in csv format, with a header line:  
Header: classPosition,#features,#classes,seperator. e.g.  
`classLast,1024,20,comma (from COIL20.data)`
* Most GP parameters are configured in gpmalmo/rundata.py
