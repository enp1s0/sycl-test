CXX=/opt/intel/oneapi/compiler/2024.0/bin/compiler/clang++
CXXFLAGS=-std=c++17 -Wall -fsycl 

TARGET=vecadd.out

$(TARGET):main.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
