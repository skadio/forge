# Forge: Foundational Optimization Embeddings From Graph Embeddings
Forge is a research library designed for representational learning in combinatorial problems. 

## Quick Start
```python
# Import Forge Library
from forge.embeddings import Forge

# Data
input_folder = "/data/"

# Run
forge = Forge(train_config_file_path="forge/configs/train_config.yaml")
forge.train(input_folder=input_folder, output_folder="/models/forge_pretrained.pkl")

# Inference
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