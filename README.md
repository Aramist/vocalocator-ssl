# Vocal Call Locator 2.0

## Installation

### Pip:
1. Clone this repo:  `git clone https://github.com/Aramist/vocalocator-ssl.git && cd vocalocator-ssl`
2. (Optional) Create a virtual environment: `python -m venv vcl_env && source vcl_env/bin/activate`
3. Install: `pip install -e .`. The `-e` option ensures that if you pull from this repo or make your own changes in the future, the changes are automatically picked up by python.

## Quick Start
This section will give you a brief intro on performing inference via a pretrained model with VCL2.
### Setup
1. Prepare your datasets according to the format below. The datasets may exist in multiple files (i.e. one dataset per recording session) as long as they reside in the same directory.
2. Unzip your pretrained model. This should yield a directory (we will refer to this as `model_dir` going forward)
3. This directory should contain, among other things, a `calibration_results.npz` file which we will use later when computing assignments.

### Training
1.  todo
### Calibration
Pretrained models should be distributed pre-calibrated, but if they are not, calibration can be done by evaluating the model with the `--calibrate` flag:
1. Prepare a dataset that is disjoint with the original training set. If the original training data is available, this can be done by leveraging the automatically generated test index at `model_dir/indices/test_set.npz`.
2. Run evaluation with the `--calibrate` flag: `python -m vocalocatorssl --data path/to/datasets/ --save-path path/to/model_dir --calibrate -o calibration_results.npz`
### Inference
Now that we have both trained weights and calibration data, we can start predicting and assigning vocalizations. The first step, *inference*, involves processing each vocalization to produce scores representing how strongly the model believes the vocalization was produced be each animal. The second step, *assignment*, takes these scores, rescoles their confidence according to the calibration data, and then assigns each vocalization to an animal.
1. Running inference with the `--predict` flag: `python -m vocalocatorssl --data path/to/datasets/ --save-path path/to/model_dir --predict -o model_predictions.npz`
2. Running assignment with the `vocalocatorssl.assign` script: `python -m vocalocatorssl.assign model_predictions.npz --calibration-results path/to/model_dir/calibration_results.npz`

The assignment script modifies the `model_predictions.npz` in-place, creating arrays by the name `{dataset_name}-assignments` within the archive. These can be accessed like any other `npz` array by unzipping the file and loading each assignment array as a `npy`. The assignments array is an integer array of shape `(num_vocalizations, )` containing `-1` for unassignable vocalizations and the predicted animal's index within the dataset's `locations` array otherwise.
```python
import numpy as np

archive = np.load('model_predictions.npz')
assn_files = list(filter(lambda k: k.endswith('assignments'), archive.files))
print(assn_files)
print(archive[assn_files[0]])
```
Out:
```
['dyad_20230207_213540-assignments', 'dyad_20250211_121652-assignments', 'dyad_20250514_181431-assignments', 'dyad_20230119_150148-assignments', 'dyad_20240830_125600-assignments', 'dyad_20250402_150005-assignments', 'dyad_20250506_155030-assignments', ...]
[0 0 0 ... 0 0 0]
```


## Advanced Usage
1. Create a dataset. This should be an HDF5 file with the following datasets:

| Dataset group/name | Shape             | Data type | Description                                                                                                                                    |
|--------------------|-------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------|
| /audio     | (*, n_channels) | float     | All sound events concatenated along axis 0                                                                                                     |
| /length_idx        | (n + 1,)                  | int       | Index into audio dataset. Sound event `i` should span the half open interval [`length_idx[i]`, `length_idx[i+1]`) and the first element should be 0. |
| /locations         | (n, num_animals, num_nodes, num_dims)  | float     | Locations associated with each sound event. Only required for training. May contain multiple nodes. Expects the origin to lie at the center of the arena.|
| /node_names        | (num_dims,)               | str(bytes)       | Names of nodes contained by `locations`.                                                                                                                                                                                 |
1. Create a config. This is a JSON file consisting of a top-level object whose properties correspond to the hyperparameters of the model and optimization algorithm. See `good_config.json` in the root directory of the repository for a good starting point.
2. Train a model: `python -m vocalocatorssl --data /path/to/directory/containing/datasets/ --config /path/to/config.json --save-path /path/to/model_dir/`
3. Using the trained model, perform calibration and inference, as shown in the quick start above

## Public datasets
See our [dataset website](https://vclbenchmark.flatironinstitute.org) to learn more about and download our public datasets.
