import pandas as pd
from .._data_export import sample_stratified

# def test_stratified_within_random_no_id_col_does_not_crash():
#     df = pd.DataFrame({"g":[0,0,1,1], "x":[10,11,12,13]})
#     out = sample_stratified(df, n=2, id_col=None, strata_cols=["g"], within="random", seed=7)
#     assert len(out) == 2
