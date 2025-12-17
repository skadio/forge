# Forge Train and Test Datasets 
Forge datasets are stored in [Hugging Face Forge Dataset](https://huggingface.co/datasets/skadio/forge). 

You can download and unzip the dataset from Hugging Face using the `/data/download.py` script. 

Each `/data/*.txt` file lists the names of the instances included in a specific task:

* pretrain.txt
  * test_pretrain.txt
* fine_tune_integral_gap.txt
  * test_integral_gap.txt
* fine_tune_variable_proba.txt
  * test_variable_proba.txt

The ICLR'26 paper experiments are denoted with prefix `iclr26_`:

* iclr26_pretrain.txt
  * iclr26_test_pretrain.txt
* iclr26_fine_tune_integral_gap.txt
  * iclr26_test_integral_gap.txt
* iclr26_fine_tune_variable_proba.txt
  * iclr26_test_variable_proba.txt