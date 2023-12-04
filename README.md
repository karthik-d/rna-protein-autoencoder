# protein-rna-mapper

## Data availability

See [LINK](https://www.dropbox.com/scl/fo/fs29ctkjojo3dd1ntagpa/h?rlkey=nebpsackstvacdw052r4nj9nx&dl=0) 


## Steps to run parameter search.

- Set the parameters to be swept in the header lines of [param_search.py](./src/param_search.py).
- Specify the data version in the calls to `CovidDataset` in function `get_data_loaders`.
- Run the script: [param_search.py](./src/param_search.py).

> [!Note]
> Currently, the default layer of the ANN data structure is loaded. Needs to be adapted to load the appropriate layer, based on `input_type`. 


## Setting training and data parameters.

### Output layer activation.

Set the parameter `output_activation` in `AutoEncoder()` call to one of:
- Linear: `'linear'`
- Sigmoid: `'sigmoid'`

### Input data normalization.

Set the parameter `normalization_method` in `CovidDataset()` call to one of:
- MinMax Linear Scaling between 0 and 1: `'minmax'`
- No normalization: `None`

During parameter search, the `normalization_method` can be set through the utility function `get_data_loaders` in [param_search.py](src/param_search.py).

> [!Note]
> Normalization is performed on `train` split only; the same normalization parameters are use to transform the `test` and `valid` splits.

### Input data type.

Set the parameter `input_type` in `CovidDataset()` call to one of:
- Log normalized expression: `'norm'`
- Raw expression: `'raw'`

During parameter search, the `input_type` can be set through the utility function `get_data_loaders` in [param_search.py](src/param_search.py).


-- this is to test if I can upload
