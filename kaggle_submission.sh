# download competition data
kaggle competitions download -c nbme-score-clinical-patient-notes

# to make submission use this command
kaggle competitions submit -c nbme-score-clinical-patient-notes -f [FILE PATH].

# to create a new dataset use the below command to generate a metadata file
kaggle datasets init -p /path/to/dataset

# add your datasets metadata to the generated file, datapack.json then run this command to create the dataset
kaggle datasets create -p /path/to/dataset

# create and run a notebook on kaggle
kaggle kernels push -k [KERNEL] -p /path/to/kernel

# download code files and metadata associated with a Notebook
kaggle kernels pull [KERNEL] -p /path/to/download -m

# retrieve a notebooks output

# get the status of the latest notebook run

# pull a notebook

# push a notebook


