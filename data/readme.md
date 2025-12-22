# Forge Train and Test Datasets 
Forge datasets are stored in [Hugging Face Forge Dataset](https://huggingface.co/datasets/skadio/forge). 

You can download and unzip the dataset from Hugging Face using the `/data/download.py` script. 
Our instances consist of MIPLIB, D-MIPLIB, StrIPlib. 

Each `/data/configs/*.txt` file lists the names of the instances included in a specific task:
* all.txt
  * This file contains all MIP instances from dmiplib, miplib and strIPlib.
* all_dmiplib.txt
  * D-MIPLIB has ~800 training instances per problem type-difficulty pair
  * These instances are the first 100 instances from the train directory for each problem type in D-MIPLIB.
  * These instances span 31 problem type-difficulty pairs
* all_miplib.txt
  * These are ~1000 instances from MIPLIB 2017. 
* all_striplib.txt
  * These are ~1700 strIPlib instances
  * These instances span 10 problem types. 
    
## OUTDATED!! 
* pretrain.txt
  This file contains all MIP instances. Specifically, it contains
  * ~1000 MIPLIB instances, sorted by size to remove any instances greater than 20 MB in the compressed format. 
  * ~3100 D-MIPLIB instances
    * These instances span 31 problem type-difficulty pairs
    * D-MIPLIB has ~800 training instances per problem type-difficulty pair
    * These instances are the first 100 instances from the train directory for each problem type in D-MIPLIB.
  * ~1700 strIPlib instances
    * These instances span 10 problem types. These are only used in pretraining. 
  * test_pretrain.txt
    * This file contains ~2950 instances across 30 problem type-difficulty pairs in D-MIPLIB.
    * These instances are drawn from the validation directory for each problem type in D-MIPLIB.
* fine_tune_integral_gap.txt
  * These instances span 30 problem type-difficulty pairs from D-MIPLIB.
  * They are the next 100 training instances. 
  * test_integral_gap.txt
    * These instances span 32 problem type-difficulty pairs from D-MIPLIB.
    * These are all of the instances from the test directory of D-MIPLIB. 


The ICLR'26 paper experiments are denoted with prefix `iclr_`:

* iclr26_pretrain.txt
  * These are ~600 MIPLIB instances sorted by size (ascending). 
  * iclr26_test_pretrain.txt
    * These instances span 22 problem type-difficulty pairs from D-MIPLIB.
    * These instances are drawn from the validation directory for each problem type in D-MIPLIB.
* iclr26_fine_tune_integral_gap.txt
  * 50 instances each of CA (very-easy, easy, medium), SC (easy, medium, hard), and GISP (easy,medium, hard) for a total of 450 training instances.
  * iclr26_test_integral_gap.txt
    * 100 instances each of very-hard CA, SC, GISP and MVC
* iclr26_fine_tune_variable_proba.txt
  * 100 instances each of CA (easy, medium), SC (easy, medium, hard) and GISP (easy, medium) for a total of 700 training instances. 
  * iclr26_test_variable_proba.txt
    * 100 instances each of medium CA, SC, GISP and MVC.

# Sort file content respecting numbers in filenames
 sort -V -o all.txt all.txt
 
# iclr_test_clusters.txt
1000 DMIPLIB instances, 50 each, from: 
CA-easy
CA-medium
CA-very-hard
CA-very-hard2
GISP-easy
GISP-medium
GISP-hard
GISP-very-hard
IP-very-hard
MIS-easy
MIS-medium
MIS-very-hard
MVC-easy
MVC-medium
MVC-hard
MVC-very-hard
SC-easy
SC-medium
SC-hard
SC-very-hard