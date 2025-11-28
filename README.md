# Forge: Foundational Optimization Embeddings From Graph Embeddings
Forge is a research library designed for representational learning in combinatorial problems. 

## Quick Start - Pretraining Pipeline
```python
from forge.embeddings import Forge
from forge.pipeline import pretrain

# Forge model
forge = Forge(train_config_yaml="forge/configs/train_config.yaml")

# Pretrain forge
pretrain(forge,
         input_mip_folder="/data/train/", 
         relaxation_list=[0.05, 0.01],
         output_mip_to_mipinfo_pkl="/models/mip_to_mipinfo.pkl",
         output_forge_pkl="/models/forge_pretrained.pkl",
         output_log_file="/models/forge_pretrained.log")
```

## Quick Start - Pretraining Script
```bash
cd forge
python -m forge.scripts.pretrain --train_config_yaml `/forge/configs/train_config.yaml` --input_mip_folder `/data/train/` --relaxation_list 0.05 0.01 --output_mip_to_mipinfo_pkl `/models/mip_to_mipinfo.pkl` --output_forge_pkl `/models/forge_pretrained.pkl` --output_log_file `/models/forge_pretrained.log`
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