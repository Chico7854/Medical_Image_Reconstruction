CXX = g++
CXXFLAGS = -O2 -std=c++17 -I/usr/include/eigen3 -I/usr/include/nlohmann -I.
LDFLAGS = -lpthread -lssl -lcrypto

all: cgnr_server

cgnr_server: server.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) server.cpp -o cgnr_server

clean:
	rm -f cgnr_server