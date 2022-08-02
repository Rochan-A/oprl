import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import sys



def _tokenizer(fname):
	with open(fname) as f:
		chunk = []
		for line in f:
			if 'START'in line:
				continue
			if 'END' in line:
				yield chunk
				chunk = []
				continue
			chunk.append(line)
	return chunk


def load_map(path):
	"""Load world from file located at path"""
	map, transition = [np.loadtxt(A) for A in _tokenizer(path)]
	map = np.array(map, dtype=np.int64)
	return map


env = load_map(sys.argv[1])/10.

clist = [
	(0/10., "gray"),
	(1/10., "black"),
	(2/10.,"purple"),
	(3/10., "red"),
	(5/10., "yellow"),
	(8/10.,"green"),
	(9/10., "red"),
	(10/10., "pink")
]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("name", clist)

matplotlib.image.imsave('{}.png'.format(sys.argv[2]), env, cmap=cmap)