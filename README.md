# RNA-Protein AutoEncoder

## Quick links.

- [Dataset [Dropbox]](https://www.dropbox.com/scl/fo/fs29ctkjojo3dd1ntagpa/h?rlkey=nebpsackstvacdw052r4nj9nx&dl=0).
- [Project Summary Report [PDF]](https://drive.google.com/file/d/1pNfT5tZM-2ehQ_AEAg_oz5HVWs1J4Gv2/view?usp=share_link).
- [Working Notes [GDocs]](https://docs.google.com/document/d/1-XD7inw_0Rk44fflm1NWWiquGqu5W8euWLPiNNtIWKU/edit?usp=sharing).
- [Shared Storage [GDrive]](https://drive.google.com/drive/folders/1Wq5CrreWzERfZCW07whHYdMR8-1UjeLq?usp=share_link).

## Steps to run parameter search.

- Set the parameters to be swept in the header lines of [param_search.py](./src/param_search.py).
- Specify the data version in the calls to `CovidDataset` in function `get_data_loaders`.
- Run the script: [param_search.py](./src/param_search.py).

## Setting training and data parameters.

### Output layer activation.

Set the parameter `output_activation` in `AutoEncoder()` call to one of:
- Linear: `'linear'`
- Sigmoid: `'sigmoid'`

### Protein data normalization.

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
