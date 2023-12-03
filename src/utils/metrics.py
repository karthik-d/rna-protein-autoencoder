import numpy as np
from scipy import stats


# assumes that the arguments are 2D matrices - samples x features.
# returns correlation between col i of arg1 and arg2.
def compute_colwise_correlations(truths, predictions):

	correlations = np.zeros((truths.shape[1]))
	for col_idx in range(truths.shape[1]):
		correlations[col_idx], _ = stats.pearsonr(
			truths[:, col_idx],
			predictions[:, col_idx]
		)

	return correlations