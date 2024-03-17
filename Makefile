CMAKE_ARGS = -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

opencv:
	mkdir -p opencv && cd opencv
	wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
	unzip opencv.zip
	mkdir -p build && cd build
	cmake ../opencv-4.x
	cmake --build .

cmake: opencv
	mkdir -p cmake-build-debug
	cd cmake-build-debug && cmake .. $(CMAKE_ARGS)

clean:
	rm -rf cmake-build-debug
.PHONY: clean
