fsvm_path=lib/github.com/fzzle/futhark-svm

.PHONY: build install clean test

build:
	futhark pkg sync
	mkdir -p python/bin
	futhark opencl --library $(fsvm_path)/main.fut -o python/bin/fsvm

install:
	cd python; pip install --user .

clean:
	find lib/github.com/diku-dk -delete
	find python/bin -delete

test:
	futhark test --backend=opencl tests/
	find tests -name '*.c' -type f -delete
	find tests ! -name '*.*' -type f -delete
