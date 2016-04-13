CC := g++ -std=c++0x

INCLUDES := -framework OpenCL

FLAGS := 

all: test 2_test

test: test.cpp
	$(CC) $(INCLUDES) $(FLAGS) -o $@ $<

2_test: 2_test.cpp
	$(CC) $(INCLUDES) $(FLAGS) -o $@ $<

clean:
	rm test 2_test
