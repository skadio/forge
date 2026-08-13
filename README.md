[![PyPI version](https://img.shields.io/pypi/v/forge-mip.svg?cacheSeconds=1)](https://pypi.org/project/forge-mip/)
[![PyPI license](https://img.shields.io/pypi/l/forge-mip.svg?cacheSeconds=1)](https://pypi.python.org/pypi/forge-mip/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Downloads](https://static.pepy.tech/personalized-badge/forge-mip?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/forge-mip)

---

<div align="center"><a name="menu"></a>
  <h3>
    <a href="https://github.com/skadio/forge?tab=readme-ov-file#quick-start">Quick-Start </a> •
    <a href="https://github.com/skadio/forge?tab=readme-ov-file#available-functionality">Available Functionality </a> •
    <a href="https://github.com/skadio/forge?tab=readme-ov-file#installation">Installation</a>
  </h3>
</div>

---

# Forge: Foundational Optimization Representations From Graph Embeddings
[Forge](https://skadio.github.io/forge/) is a research library designed for representational learning in combinatorial optimization. It provides tools for generating embeddings from MIP instances, pre-training models on these embeddings, and fine-tuning them for specific tasks such as predicting integral gap, search guidance, backdoor prediction, and solver configuration.

## Quick Start
```bash
# Install the library
pip install forge-mip

# Generate MIP Embeddings from the Hugging Face pre-trained Forge model and save the output
# Export your Hugging Face Token, if not already set in your environment
# export HF_TOKEN=<your_hugging_face_token>
forge --input_mips ./data/instances/ --input_mip_instances_file ./data/configs/test.txt --output_mip_to_embeddings_pkl ./models/mip_to_embeddings.pkl

# Generate MIP Embeddings from a local pre-trained Forge model and save the output
forge --train_config_yaml ./forge/configs/train_config.yaml --input_forge_pkl ./models/forge_pretrained.pkl --input_mips ./data/instances/ --input_mip_instances_file ./data/configs/test.txt --output_mip_to_embeddings_pkl ./models/mip_to_embeddings.pkl
```

**Note:** When no local model is found, `forge` downloads the pre-trained Forge model from Hugging Face and saves it to the package's models directory — for a `pip install`-ed package that is `<site-packages>/models/forge_pretrained.pkl`, not `./models/`. To use a model stored in the repo, place it at `./models/forge_pretrained.pkl` and pass `--input_forge_pkl ./models/forge_pretrained.pkl`.

**Access Request:** The [pretrained Forge model](https://huggingface.co/skadio/forge) is gated, please request access first. 
Without an HF token with approved access, the load from Hugging Face will fail.

## Available Functionality

| Functionality                                                             | Description                                                                                      |
|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| [Generate MIP Info](#generate-mip-info)                                | Build and serialize MIPInfo objects from raw MIP instances for reuse in downstream pre-training. |
| [Pre-Train Forge](#pre-train-embeddings)                                  | Pre-train Forge on MIP instances and their MIPInfo and save a pretrained model checkpoint.       |
| [Generate Embeddings](#generate-embeddings)                              | Generate per-instance embeddings from a pretrained Forge model (MIPEmbeddings).                  |
| [Fine-Tune Integral Gap](#fine-tune-integral-gap)       | Fine-tune Forge for integral-gap prediction on labeled GapInfo data.                             |
| [Predict Integral Gap](#predict-integral-gap)           | Run inference with a fine-tuned model to predict LP/MIP gap information (GapInfo).               |
| [Fine-Tune Variable Probabilities](#fine-tune-variable-probabilities)     | Fine-tune Forge for variable 0/1 probability prediction on labeled TripletInfo data.             |
| [Predict Variable Probabilities](#predict-variable-probabilities)         | Run inference with a fine-tuned model to predict variable 0/1 probabilities (HintInfo).          |

## Generate MIP Info

```python
from forge.embeddings import Forge
from forge.pipeline import mip_to_mipinfo
from forge.utils import Constants

# Forge model with its pre-trained configuration
forge = Forge(train_config_yaml="./forge/configs/train_config.yaml")

# Generate MIP info object for a set of given mip instances 
# The output mip_to_mipinfo pickle is stored as output_mip_to_mipinfo_pkl
# The mip_to_mipinfo pkl can be re-used in pretrain() with input_mip_to_mipinfo_pkl flag
#   - mip_to_mipinfo maps mip instance to a mipinfo object, Dict[str, MIPInfo], containing:
#       - instance_name: str, the name of the MIP instance
#       - feature_tensor: torch.Tensor, the feature tensor for the MIP instance (num_cons + num_vars, feat_dim=10)
#       - num_cons: int, the number of constraints in the MIP instance
#       - num_vars: int, the number of variables in the MIP instance
#       - edge_index: torch.Tensor, (2, E) edges from source (constraint) to target (variable) nodes 
#       - edge_weight: torch.Tensor, (E,), weights of the edges
# Pretraining log is stored in output_log_file with loss curves and training details
mip_to_mipinfo(forge=forge,
               input_mip_folder="./data/instances/",
               input_mip_instances_file="./data/configs/test.txt",
               output_mip_to_mipinfo_pkl="./models/test_mip_to_mipinfo.pkl",
               relaxation_list=[0.05, 0.01],
               num_parallel_workers=1)
```
##### Command Line
```bash
cd forge
python -m scripts.mip_to_mipinfo --train_config_yaml ./forge/configs/train_config.yaml --input_mip_folder ./data/instances/ --input_mip_instances_file ./data/configs/all.txt --output_mip_to_mipinfo_pkl ./models/mip_to_mipinfo.pkl --relaxation_list 0.05 0.01 --num_parallel_workers 1
```

## Pre-Train Embeddings

```python
from forge.embeddings import Forge
from forge.pipeline import pretrain

# Forge model with its pre-training configuration
forge = Forge(train_config_yaml="./forge/configs/train_config.yaml")

# Pretrain Forge on a set of MIP instances in the given input folder
# The pretrained model pickle is stored as output_forge_pretrained_pkl
# The intermediate mip_to_mipinfo pickle is stored as output_mip_to_mipinfo_pkl
# The mip_to_mipinfo pkl can be reused with input_mip_to_mipinfo_pkl flag to skip MIP parsing in future pre-training
#   - mip_to_mipinfo maps mip instance to a mipinfo object, Dict[str, MIPInfo], containing:
#       - instance_name: str, the name of the MIP instance
#       - feature_tensor: torch.Tensor, the feature tensor for the MIP instance (num_cons + num_vars, feat_dim=10)
#       - num_cons: int, the number of constraints in the MIP instance
#       - num_vars: int, the number of variables in the MIP instance
#       - edge_index: torch.Tensor, (2, E) edges from source (constraint) to target (variable) nodes 
#       - edge_weight: torch.Tensor, (E,), weights of the edges
# Pretraining log is stored in output_log_file with loss curves and training details
pretrain(forge=forge,
         input_mip_folder="./data/instances/",
         input_mip_instances_file="data/configs/all.txt",
          output_mip_to_mipinfo_pkl="./models/pretrain_clusters_mip_to_mipinfo.pkl",
         output_forge_pretrained_pkl="./models/forge_pretrained.pkl",
         output_log_file="./models/forge_pretrained.log")
```

##### Command Line
```bash
cd forge
python -m scripts.pretrain --train_config_yaml ./forge/configs/train_config.yaml --input_mip_folder ./data/instances/ --input_mip_instances_file ./data/configs/all.txt --relaxation_list 0.05 0.01 --output_mip_to_mipinfo_pkl ./models/pretrain_clusters_mip_to_mipinfo.pkl --output_forge_pretrained_pkl ./models/forge_pretrained.pkl --output_log_file ./models/forge_pretrained.log
```

## Generate Embeddings

```python
from forge.embeddings import Forge
from forge.pipeline import mip_to_embeddings
from forge.utils import Constants

# Forge model with its pre-trained configuration
forge = Forge(train_config_yaml="./forge/configs/train_config.yaml")

# Generate embeddings dictionary for MIPs in the input folder
# Use the trained Forge model stored in input_forge_pkl of type model_type
# The output mip_to_embeddings pickle is stored as output_mip_to_embeddings_pkl
#   Each MIP instance is mapped to a MIPEmbeddings object, Dict[str, MIPEmbeddings], containing: 
#       - instance_embedding: np.ndarray (codebook_size)
#       - embedding_of_constraint: torch.Tensor (num_constraints, codebook_dim)
#       - embedding_of_variable: torch.Tensor (num_variables, codebook_dim)
mip_to_embeddings_dict = mip_to_embeddings(forge=forge,
                                           input_mips="./data/instances/",
                                           input_mip_instances_file="./data/configs/test.txt",
                                           input_forge_pkl="./models/forge_pretrained.pkl",
                                           model_type=Constants.FORGE_PRE_TRAIN,
                                           instance_embedding_only=False,
                                           output_mip_to_embeddings_pkl="./models/mip_to_embeddings.pkl")
```
##### Command Line
```bash
cd forge
python -m scripts.mip_to_embeddings --train_config_yaml ./forge/configs/train_config.yaml --input_forge_pkl ./models/forge_pretrained.pkl --input_mips ./data/instances/ --input_mip_instances_file ./data/configs/test.txt --output_mip_to_embeddings_pkl ./models/mip_to_embeddings.pkl
```

## Fine-Tune Integral Gap

```python
from forge.embeddings import Forge
from forge.pipeline import finetune_integral_gap
from forge.utils import Constants

# Forge model with its pre-trained configuration
forge = Forge(train_config_yaml="./forge/configs/train_config.yaml")

# Fine-tune Forge to predict integral gaps
finetune_integral_gap(forge=forge,
                      input_forge_pkl="./models/forge_pretrained.pkl",
                      model_type=Constants.FORGE_FINE_TUNE_INTEGRAL_GAP,
                      input_mip_folder="./data/instances/",
                      input_mip_instances_file="data/configs/fine_tune_integral_gap.txt",
                      output_forge_finetuned_pkl="./models/forge_integral_gap.pkl",
                      output_mip_to_gapinfo_pkl="./models/mip_to_gapinfo.pkl",
                      num_parallel_workers=5)
```

##### Command Line
```bash
cd forge
python -m scripts.finetune_integral_gap --train_config_yaml ./forge/configs/train_config.yaml --input_forge_pkl ./models/forge_pretrained.pkl --input_mip_folder ./data/instances/ --input_mip_instances_file ./data/configs/fine_tune_integral_gap.txt --output_forge_finetuned_pkl ./models/forge_integral_gap.pkl --output_mip_to_gapinfo_pkl ./models/mip_to_gapinfo.pkl
```

## Predict Integral Gap

```python
from forge.embeddings import Forge
from forge.pipeline import mip_to_gapinfo
from forge.utils import Constants

# Forge model with its pre-trained configuration
forge = Forge(train_config_yaml="./forge/configs/train_config.yaml")

# Predict integral gaps
# Each MIP instance is mapped to a GapInfo object, Dict[str, GapInfo], containing:
#   - lp_obj: the true objective value of the lp relaxation solution
#   - lp_sol: the true lp relaxation solution
#   - mip_obj: the predicted objective value of the mip solution
#   - mip_sol: None, there is no solution, only gap prediction
#   - gap_ratio: float, the predicted ratio between lp and mip 
mip_to_gapinfo_dict = mip_to_gapinfo(forge=forge,
                                     input_forge_pkl="./models/forge_integral_gap.pkl",
                                     model_type=Constants.FORGE_FINE_TUNE_INTEGRAL_GAP,
                                     input_mips="./data/instances/",
                                     input_mip_instances_file="./data/configs/test_integral_gap.txt",
                                     output_mip_to_gapinfo_pkl="./models/mip_to_gapinfo.pkl",
                                     problem_type="CA")
```

##### Command Line
```bash
cd forge
python -m scripts.mip_to_gapinfo --train_config_yaml ./forge/configs/train_config.yaml --input_forge_pkl ./models/forge_integral_gap.pkl --input_mips ./data/instances/ --input_mip_instances_file ./data/configs/test_integral_gap.txt --output_mip_to_gapinfo_pkl ./models/mip_to_gapinfo.pkl --problem_type CA
```

## Fine-Tune Variable Probabilities

```python
from forge.embeddings import Forge
from forge.pipeline import finetune_variable_proba
from forge.utils import Constants

# Forge model with its pre-trained configuration
forge = Forge(train_config_yaml="./forge/configs/train_config.yaml")

# Fine-tune Forge to predict variable probabilities
finetune_variable_proba(forge=forge,
                        input_forge_pkl="./models/forge_pretrained.pkl",
                        model_type=Constants.FORGE_FINE_TUNE_VARIABLE_PROBA,
                        input_mip_folder="./data/instances/",
                        input_mip_instances_file="data/configs/fine_tune_variable_proba.txt",
                        output_forge_finetuned_pkl="./models/forge_variable_proba.pkl",
                        output_mip_to_tripletinfo_pkl="./models/output_mip_to_tripletinfo.pkl",
                        triplet_time_limit=300,
                        triplet_num_solutions=5)
``` 

##### Command Line
```bash
cd forge
python -m scripts.finetune_variable_proba --train_config_yaml ./forge/configs/train_config.yaml --input_forge_pkl ./models/forge_pretrained.pkl --input_mip_folder ./data/instances/ --input_mip_instances_file ./data/configs/fine_tune_variable_proba.txt --output_forge_finetuned_pkl ./models/forge_variable_proba.pkl --output_mip_to_tripletinfo_pkl ./models/output_mip_to_tripletinfo.pkl
```

## Predict Variable Probabilities

```python
from forge.embeddings import Forge
from forge.pipeline import mip_to_hint
from forge.utils import Constants

# Forge model with its pre-trained configuration
forge = Forge(train_config_yaml="./forge/configs/train_config.yaml")

# Predict variable 0/1 probabilities
# Each MIP instance is mapped to a HintInfo object, Dict[str, HintInfo], containing:
#   - hint_ones: variable indices to hint as 1
#   - hint_zeros: variable indices to hint as 0
#   - hint_pri_ones: priority ranks for the 1-hints (higher = more confident)
#   - hint_pri_zeros: priority ranks for the 0-hints (higher = more confident)
mip_to_hintinfo_dict = mip_to_hint(forge=forge,
                                   input_forge_pkl="./models/forge_variable_proba.pkl",
                                   model_type=Constants.FORGE_FINE_TUNE_VARIABLE_PROBA,
                                   input_mips="./data/instances/",
                                   input_mip_instances_file="./data/configs/test_variable_proba.txt",
                                   output_mip_to_hintinfo_pkl="./models/mip_to_hintinfo.pkl",
                                   problem_type="CA")
```

##### Command Line
```bash
cd forge
python -m scripts.mip_to_hint --train_config_yaml ./forge/configs/train_config.yaml --input_forge_pkl ./models/forge_variable_proba.pkl --input_mips ./data/instances/ --input_mip_instances_file ./data/configs/test_variable_proba.txt --output_mip_to_hintinfo_pkl ./models/mip_to_hintinfo.pkl --problem_type CA
```

## Installation
Forge requires **Python 3.10 or newer** and can be installed via `pip install forge-mip`. 

### Installation from Source Code
```
git clone https://github.com/skadio/forge.git
cd forge
pip install build # if build is not installed
python -m build
pip install dist/forge_mip-X.X.X-py3-none-any.whl
```

### Test Your Setup
```
$ git clone https://github.com/skadio/forge.git
$ cd forge
$ python -m unittest discover tests
```

## Support
Please submit bug reports and feature requests as [Issues](https://github.com/skadio/forge/issues).

## Acknowledgments
We would like to thank [Modal](https://modal.com/) for their generous support through the provision of academic credits and computational infrastructure, which were instrumental in training the Forge model used in this research.

## License
Forge is licensed under the [Apache License 2.0](LICENSE).

<br>
