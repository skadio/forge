# Forge: Foundational Optimization Embeddings From Graph Embeddings
Forge is a research library designed for representational learning in combinatorial problems. 

## Quick Start: Pre-Train Embeddings

```python
from forge.embeddings import Forge
from forge.pipeline import pretrain

# Forge model with its pre-training configuration
forge = Forge(train_config_yaml="/forge/configs/train_config.yaml")

# Pretrain Forge on a set of MIP instances
pretrain(forge=forge,
         input_mip_folder="/data/train/",
         output_mip_to_mipinfo_pkl="/models/mip_to_mipinfo.pkl",    # Bipartite graph and mip metadata
         output_forge_pretrained_pkl="/models/forge_pretrained.pkl",
         output_log_file="/models/forge_pretrained.log")
```

##### Command Line
```bash
cd forge
python -m forge.scripts.pretrain --train_config_yaml `/forge/configs/train_config.yaml` --input_mip_folder `/data/train/` --relaxation_list 0.05 0.01 --output_mip_to_mipinfo_pkl `/models/mip_to_mipinfo.pkl` --output_forge_pkl `/models/forge_pretrained.pkl` --output_log_file `/models/forge_pretrained.log`
```

## Quick Start: Generate Embeddings
```python
from forge.embeddings import Forge
from forge.pipeline import mip_to_embeddings

# Forge model with its pre-trained configuration
forge = Forge(train_config_yaml="/forge/configs/train_config.yaml")

# Load pre-trained model
forge.load_model(input_forge_pkl="/models/forge_pretrained.pkl")

# Generate embeddings dictionary for MIPs in the input folder
# Each MIP instance is mapped to a MIPEmbeddings object, Dict[str, MIPEmbeddings], containing: 
#   - instance_embedding: np.ndarray (codebook_size)
#   - embeddings_of_constraint[c]: torch.Tensor(num_constraints, codebook_dim)
#   - embeddings_of_variable[v]: torch.Tensor(num_constraints, codebook_dim) 
mip_to_embeddings_dict = mip_to_embeddings(forge=forge,
                                           input_mips="/data/test/",
                                           output_mip_to_embeddings_pkl="/models/mip_to_embeddings.pkl")
```
##### Command Line
```bash
cd forge
python -m forge.scripts.mip_to_embeddings --train_config_yaml `/forge/configs/train_config.yaml` --input_forge_pkl `/models/forge_pretrained.pkl` --input_mips `/data/test/` --output_mip_to_embeddings_pkl `/models/mip_to_embeddings.pkl`
```

## Quick Start: Fine-Tune Integral Gap

```python
from forge.embeddings import Forge
from forge.pipeline import finetune_integral_gap
from forge.utils import Constants

# Forge model with its pre-trained configuration
forge = Forge(train_config_yaml="/forge/configs/train_config.yaml")

# Load pre-trained model ready for integrality gap 
forge.load_model(input_forge_pkl="/models/forge_pretrained.pkl", model_type=Constants.FORGE_FINE_TUNE_INTEGRAL_GAP)

# Fine-tune Forge to predict integral gaps
finetune_integral_gap(forge=forge,
                      input_mip_folder="/data/train/",
                      output_forge_finetuned_pkl="/models/forge_integral_gap.pkl",
                      output_mip_to_gapinfo_pkl="/models/mip_to_gapinfo.pkl")
```

##### Command Line
```bash
cd forge
python -m forge.scripts.finetune_integral_gap --train_config_yaml `/forge/configs/train_config.yaml` --input_forge_pkl `/models/forge_pretrained.pkl` --input_mip_folder `/data/train/` --output_forge_finetuned_pkl `/models/forge_integral_gap.pkl` --output_mip_to_gapinfo_pkl `/models/mip_to_gapinfo.pkl`
```

## Quick Start: Predict Integral Gap

```python
from forge.embeddings import Forge
from forge.pipeline import mip_to_gap_info
from forge.utils import Constants

# Forge model with its pre-trained configuration
forge = Forge(train_config_yaml="/forge/configs/train_config.yaml")

# Load pre-trained model ready for integrality gap
forge.load_model(input_forge_pkl="/models/forge_integral_gap.pkl", model_type=Constants.FORGE_FINE_TUNE_INTEGRAL_GAP)

# Predict integral gaps
# Each MIP instance is mapped to a GapInfo object, Dict[str, GapInfo], containing:
#   - ratio: float, the predicted ratio between lp and mip
#   - mip_sol: None, there is no solution prediction, just the gap
#   - mip_obj: the predicted objective value of the mip solution 
#   - lp_obj: the true objective value of the lp relaxation solution 
mip_to_gap_info_dict = mip_to_gap_info(forge=forge,
                                       input_mips="/data/test/",
                                       output_mip_to_gap_info_pkl="/models/mip_to_gapinfo.pkl",
                                       problem_type="SC")
```

##### Command Line
```bash
cd forge
python -m forge.scripts.mip_to_gap_info --train_config_yaml `/forge/configs/train_config.yaml` --input_forge_pkl `/models/forge_integral_gap.pkl` --input_mips `/data/test/` --output_mip_to_gap_info_pkl `/models/mip_to_gapinfo.pkl` --problem_type SC
```

## Installation
Forge requires **Python 3.10** and can be installed via `pip install forge`. 

As the backbone for graph neural networks, Forge uses [VQGraph](https://github.com/YangLing0818/VQGraph).

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