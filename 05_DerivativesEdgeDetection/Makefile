CC = g++
CFLAGS = -Wall -O2 -std=c++17
LDFLAGS = `pkg-config --cflags --libs opencv4`

SRC_FILES = $(shell find . -type f -name "*.cpp")
TARGETS = $(SRC_FILES:%.cpp=%)

all: check_opencv $(TARGETS)

$(TARGETS): % : %.cpp
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

check_opencv:
	@if ! pkg-config --exists opencv4; then \
		echo "Error: OpenCV library not found. Please install libopencv-dev."; \
		exit 1; \
	fi

clean:
	rm -f $(TARGETS)
