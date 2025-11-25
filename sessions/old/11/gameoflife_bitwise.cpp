#include <iostream>
using namespace std;

/*
     00000000000000000000000000000000000000000000000000
     00000000000000000000000000000000000000000000000000
     00000000000000000000000000000000000000000000000000


		 case 1: if I, my neighbors to north, and south are all zero,
		 then next time I am still all zero


     00000000000000000000000000000000000000000000000000
     00000000000000000000000000000000000000000000000111
     00000000000000000000000000000000000000000000000000

		 spread the bits
                                    0  0  0  0  0  0  0
                                    0000000000000000000
                                    0  0  0  0  1  1  1
                                    0000000000000001000

				1111         111
        1yx0         111
        1000         111


                     abc
                      x
     NORTH, SOUTH, ME

     bitpos(int i, uint64_t N, uint64_t ME, uint64_t S) {
		 int count = 0;
		 if (N & (1ULL << i))
   		 count++;
		 if (N & (1ULL << (i-1)))
   		 count++;
		 if (N & (1ULL << (i+1)))
   		 count++;
		 if (S & (1ULL << i))
   		 count++;
		 if (S & (1ULL << (i-1)))
   		 count++;
		 if (S & (1ULL << (i+1)))
   		 count++;
		 if (ME & (1ULL << (i+1))
		   count++:
		 if (ME & (1ULL << (i-1))
		   count++:


      111  101 001


			mask = 7;
			count = popcount(N & mask) + popcount(ME & mask) + popcount(S & mask);
0b111) //   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx101
      popcount(v & (0b111 << 1)
         
 in verilog it might look like this
     input [63:0] a_in,
     input [63:0] b_in;
     input [63:0] c_in;

     for (int i = 0; i < 64; i++) {
		   count[i] = a_in[i] + a_in[i-1] + 
       if (count[i] < 2 || count[i] > 3) TODO: logic is flawed. different for dead/alive
         out[i] = 0; // dead




 */
 

uint32_t n;
uint64_t* life;
uint64_t* nextlife;

void setLife(const int i) {
    life[i/64] |= (1ULL << (i% 64));
}
bool isAlive(const int i) {
    return life[i/64] &= (1ULL << (i% 64));
}

void init() {
    const uint32_t size = (n+2)*(n+2);
    const uint32_t word_size =(size+63)/64; 
    life = new uint64_t[word_size];
    nextlife = new uint8_t[word_size];

    for (auto i = 0; i < word_size; i++)
      life[i] = 0;
    setLife(100);
    setLife(101);
    setLife(102);
}



/*
    vector approach

    10010001


       1    0    0    1    0    0    0    1
    0000 0000 0000 0000 0000 0000 0000 0000

    00000000000000000000000000000000
    00000000000000000000110000000000
    00000000000000000000110000000000
    00000000000000000000000000000000


*/

void calcLiveOrDead(uint32_t i) {
    const uint32_t EAST = +1;
    const uint32_t WEST = -1;
    const uint32_t NORTH = -n-2;
    const uint32_t SOUTH = +n+2;
     int count = isAlive(i+EAST) + isAlive(i+WEST) + // TODO: life[i+NORTH] + life[i+SOUTH] +
        life[i+NORTH+EAST] + life[i+NORTH + WEST] + life[i+SOUTH + EAST] + life[i+SOUTH+WEST];
  if (isAlive(i)) {
    //TODO: nextlife[i] = count >=2 && count <= 3;
  } else {
    //TODO: nextlife[i] = count == 3;

  }
}

void stepForward() {

    for (int i = 0, c = n+2+1; i < n; i++, c+= 2)
      for (auto j = 0; j < n ; j++, c++)
        calcLiveOrDead(c);
    swap(life, nextlife);
}

void print() {
    for (int i = 0, c = n+2+1; i < n; i++, c += 2) {
      for (auto j = 0; j < n; j++, c++)
        cout << int(life[c]) << ' ';
      cout << "\n";
    }
    cout << "\n\n";
}
int main() {
    n = 10;
    int row = n+2;
	constexpr auto num_generations = 4;
    init();
    life[2*row+3] = 1;
    life[2*row+4] = 1;
    life[2*row+5] = 1;
    for (auto i = 0; i < num_generations; i++) {
      stepForward();
      print();
    }


}
