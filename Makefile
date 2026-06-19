CXX = g++
CXXFLAGS = -O3 -std=c++17 -I/usr/include/eigen3 -lpthread

all: server

server: server.cpp
	$(CXX) $(CXXFLAGS) server.cpp -o server

clean:
	rm -rf metrics_py.csv metrics_cpp.csv relatorio_client* images_client* server