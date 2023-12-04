import matplotlib.pyplot as plot 


def save_line_plots(x_data, y_data, labels, axis_labels, savepath):

	plot.clf()
	for x, y, label in zip(x_data, y_data, labels):
		plot.plot(x, y, label=label)
	
	plot.legend()
	plot.xlabel(axis_labels['x'])
	plot.ylabel(axis_labels['y'])

	# save figure.
	plot.savefig(savepath)