# Forge: Foundational Optimization Embeddings From Graph Embeddings
Forge is a research library designed for representational learning in combinatorial problems. 

## Quick Start - Pre-Training Pipeline
```python
from forge.embeddings import Forge
from forge.pipeline import pretrain

# Forge model with its pre-training configuration
forge = Forge(train_config_yaml="/forge/configs/train_config.yaml")

# Pretrain forge
pretrain(forge=forge,
         input_mip_folder="/data/train/", 
         relaxation_list=[0.05, 0.01],
         output_mip_to_mipinfo_pkl="/models/mip_to_mipinfo.pkl",
         output_forge_pkl="/models/forge_pretrained.pkl",
         output_log_file="/models/forge_pretrained.log")
```

## Quick Start - Pre-Training Script
```bash
cd forge
python -m forge.scripts.pretrain --train_config_yaml `/forge/configs/train_config.yaml` --input_mip_folder `/data/train/` --relaxation_list 0.05 0.01 --output_mip_to_mipinfo_pkl `/models/mip_to_mipinfo.pkl` --output_forge_pkl `/models/forge_pretrained.pkl` --output_log_file `/models/forge_pretrained.log`
```

## Quick Start - MIP Embeddings Pipeline
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

## Quick Start - MIP Embeddings Script
```bash
cd forge
python -m forge.scripts.mip_to_embeddings --train_config_yaml `/forge/configs/train_config.yaml` --input_mips `/data/test/` --output_mip_to_embeddings_pkl `/models/mip_to_embeddings.pkl`
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
Forge is licensed under the [Apache License 2.0](LICENSE.md).

<br>