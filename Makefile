CMAKE_ARGS = -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

opencv:
	sh opencv.sh

cmake: opencv
	mkdir -p cmake-build-debug
	cd cmake-build-debug && cmake .. $(CMAKE_ARGS)

clean:
	rm -rf cmake-build-debug
.PHONY: clean
