# protein-rna-mapper

\\---- Data ---- \\

See [LINK](https://www.dropbox.com/scl/fo/fs29ctkjojo3dd1ntagpa/h?rlkey=nebpsackstvacdw052r4nj9nx&dl=0) 


## Steps to run parameter search.

- Set the parameters to be swept in the header lines of [param_search.py](./src/param_search.py).
- Specify the data version in the calls to `CovidDataset` in function `get_data_loaders`.
- Run the script: [param_search.py](./src/param_search.py).

> [!Note]
> Currently, the default layer of the ANN data structure is loaded. Needs to be adapted to load the appropriate layer, based on `input_type`. 
