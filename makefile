SRC_PATH=lib/github.com/fzzle/futhark-svm

clean:
	find bin/ -type f -delete
	find bench/* ! -name '*.fut' -delete

build:
	futhark opencl --library $(SRC_PATH)/main.fut -o bin/main
	cd bin; build_futhark_ffi main
