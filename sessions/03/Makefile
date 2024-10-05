OPT	:= -O2
DEBUG	:= -g
COMP	:= g++ $(OPT) $(DEBUG)

EXE	:= 00_simple_eratosthenes 01_better_eratosthenes 02_bitwise_eratosthenes 03_syncronization
all: $(EXE)

00_simple_eratosthenes: 00_simple_eratosthenes.cpp
	$(COMP) $< -o $@

01_better_eratosthenes: 01_better_eratosthenes.cpp
	$(COMP) $< -o $@

02_bitwise_eratosthenes: 02_bitwise_eratosthenes.cpp
	$(COMP) -mpopcnt $< -o $@

03_syncronization: 03_syncronization.cpp
	$(COMP) $< -o $@

# wipe it all out so you can build clean
clean:
	rm $(EXE)

n	:= 1000000000
bench:
	time ./00simple_eratosthenes $(n)
	time ./01_better_eratosthenes $(n)
	time ./02_bitwise_eratosthenes $(n)
