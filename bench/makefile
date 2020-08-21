.PHONY: mnist adult

mnist:
	pip3 install --user idx2numpy numpy sklearn
	wget -P data http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	wget -P data http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	wget -P data http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	wget -P data http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	gunzip data/t*-ubyte.gz

adult:
	wget https://datahub.io/machine-learning/adult/r/1.csv -O data/adult.csv