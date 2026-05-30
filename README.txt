pip install -r requirements.txt
python Experiment.py --subjects 1 2 3   # quick 3-subject test
python Experiment.py                     # full 9 subjects
python Experiment.py --snn_mode both    # both hybrid and full SNN will run 


snn_variants = {
    "hybrid": ["HybridSNN"],
    "full":   ["FullySNN"],
    "both":   ["HybridSNN", "FullySNN"],
}

The full subject will take signifactly longer then the 3 subject. 

When you first run either of the Experiment.py lines a folder names "C-" will be generated, this just holds the dataset so you don't have to reload it every time. A results folder will albe generated this will contain the confusion matrix, test/trainloss graphs, and energy comparisons. 

No special downloads of files are needed to make this work just install the requirements as well as spikingjelly as indicated.