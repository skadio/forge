# Forge: Foundational Optimization Embeddings From Graph Embeddings
Forge is a research library designed for representational learning in combinatorial problems. 

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
               input_mip_instances_file="./data/configs/test_pretrain.txt",
               output_mip_to_mipinfo_pkl="./models/forge_pretrained.pkl",
               relaxation_list=[0.05, 0.01])
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
         input_mip_instances_file="./data/configs/pretrain.txt",
         output_mip_to_mipinfo_pkl="./models/mip_to_mipinfo.pkl",
         output_forge_pretrained_pkl="./models/forge_pretrained.pkl",
         output_log_file="./models/forge_pretrained.log")
```

##### Command Line
```bash
cd forge
python -m scripts.pretrain --train_config_yaml ./forge/configs/train_config.yaml --input_mip_folder ./data/instances/ --input_mip_instances_file ./data/configs/pretrain.txt --relaxation_list 0.05 0.01 --output_mip_to_mipinfo_pkl ./models/mip_to_mipinfo.pkl --output_forge_pretrained_pkl ./models/forge_pretrained.pkl --output_log_file ./models/forge_pretrained.log
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
#       - embeddings_of_constraint[c]: torch.Tensor(num_constraints, codebook_dim)
#       - embeddings_of_variable[v]: torch.Tensor(num_constraints, codebook_dim) 
mip_to_embeddings_dict = mip_to_embeddings(forge=forge,
                                           input_mips="./data/instances/",
                                           input_mip_instances_file="./data/configs/test_pretrain.txt",
                                           input_forge_pkl="./models/forge_pretrained.pkl",
                                           model_type=Constants.FORGE_PRE_TRAIN,
                                           output_mip_to_embeddings_pkl="./models/mip_to_embeddings.pkl")
```
##### Command Line
```bash
cd forge
python -m scripts.mip_to_embeddings --train_config_yaml ./forge/configs/train_config.yaml --input_forge_pkl ./models/forge_pretrained.pkl --input_mips ./data/instances/ --input_mip_instances_file ./data/configs/test_pretrain.txt --output_mip_to_embeddings_pkl ./models/mip_to_embeddings.pkl
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
                      input_mip_instances_file="./data/configs/fine_tune_integral_gap.txt",
                      output_forge_finetuned_pkl="./models/forge_integral_gap.pkl",
                      output_mip_to_gapinfo_pkl="./models/mip_to_gapinfo.pkl",
                      num_parallel_workers = 5)
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
forge = Forge(train_config_yaml="/forge/configs/train_config.yaml")

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
                                     input_mip_instances_file="./data/configs/test_fine_tune_integral_gap.txt",
                                     output_mip_to_gapinfo_pkl="./models/mip_to_gapinfo.pkl",
                                     problem_type="CA")
```

##### Command Line
```bash
cd forge
python -m forge.scripts.mip_to_gapinfo --train_config_yaml ./forge/configs/train_config.yaml --input_forge_pkl ./models/forge_integral_gap.pkl --input_mips ./data/instaces/ --input_mip_instances_file ./data/configs/test_fine_tune_integral_gap.txt --output_mip_to_gap_info_pkl ./models/mip_to_gapinfo.pkl --problem_type AC
```

## Installation
Forge requires **Python 3.10** and can be installed via `pip install forge`. 

### Installation from Source Code
```
git clone https://github.com/skadio/forge.git
cd forge
pip install build # if build is not installed
python -m build
pip install dist/forge-X.X.X-py3-none-any.whl
```

### Test Your Setup
```
$ git clone https://github.com/skadio/forge.git
$ cd forge
$ python -m unittest discover tests
```

## Support
Please submit bug reports and feature requests as [Issues](https://github.com/skadio/forge/issues).

## License
Forge is licensed under the [Apache License 2.0](LICENSE).

<br>