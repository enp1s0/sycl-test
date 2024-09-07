CXX=icpx
CXXFLAGS=-std=c++17 -Wall -fsycl -qopenmp

TARGET=device.out

$(TARGET):main.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
