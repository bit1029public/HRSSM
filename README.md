This is the PyTorch implementation for our method.

## Requirements

Our code is based on [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) and the Realistic Maniskill environment is adopted from [RePo](https://github.com/zchuning/repo).

## Usage

To train the model in the paper,  please download the videos labeled 'driving_car' in the Kinetics 400 dataset and run the following command:

```bash
python -u dreamer.py --configs dmc_vision  --task dmc_walker_stand_video --seed 0 --logdir ./log
````
