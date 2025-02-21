CC = g++
CFLAGS = -Wall -O2 -std=c++17
LDFLAGS = `pkg-config --cflags --libs opencv4`
TARGET = point_filtering
SRC = 03_TransformationInCpp/point_filtering.cpp

all: check_opencv $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

check_opencv:
	@if ! pkg-config --exists opencv4; then \
		echo "Error: OpenCV library not found. Please install libopencv-dev."; \
		exit 1; \
	fi

clean:
	rm -f $(TARGET)
