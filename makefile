FSVM_PATH=lib/github.com/fzzle/futhark-svm

clean:
	find bin/ -type f -delete
	find bench/* ! -name '*.fut' -delete
	find tests/* ! -name '*.fut' -delete

setup:
	futhark pkg sync

build: # Build for use w/ Python.
	mkdir -p bin
	futhark opencl --library $(FSVM_PATH)/main.fut -o bin/main
	cd bin; build_futhark_ffi main

test:
	futhark test --backend=opencl tests/
	find tests/* ! -name '*.fut' -delete