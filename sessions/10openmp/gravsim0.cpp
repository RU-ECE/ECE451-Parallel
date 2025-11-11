#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
using namespace std;

constexpr double G = 5.67E-11; // universal gravitational constant

/*
	Gravitational force F = G m1 m2 / r^2
	Every body exerts a force on every other body.

	find the problems in this code
	compiler doesn't do well with floating point

	c1*x*c2 = (c1*c2)*x
	x*c1*c2 = x*(c1*c2)
	
	
	*/
#if 0
// suggested: Structure of arrays instead of array of structures
	class Body {
private:
	std::vector<std::string> names;
	std::vector<double> masses; // rest mass
	double x,y,z; // location
	double vx, vy, vz; // velocity
#endif

class Body {
private:
	std::string name;
	double m; // rest mass
	double x,y,z; // location
	double vx, vy, vz; // velocity
public:
	Body(string name, double m, double x, double y, double z,
			 double vx, double vy, double vz) :
		m(m), x(x), y(y), z(z), vx(vx), vy(vy), vz(vz) {}
	virtual ~Body();
	friend void step_forward(double dt);
	friend ostream& operator <<(ostream& s, const Body& b) {
		return s << setw(14) <<	b.name << setw(14) <<	b.x << setw(14) << b.y <<
			setw(14) << b.vx << setw(14) << b.vy;
	}
	virtual double dist(const Body* b) const;

	// given the acceleration, calculate the component in the x, y, z direction
	virtual double d2x(const Body* other, double a) const;
	virtual double d2y(const Body* other, double a) const;
	virtual double d2z(const Body* other, double a) const;
	virtual void step_forward(double dt);
};

Body::~Body() {
}

double Body::dist(const Body* b) const {
	double dx = this->x - b->x; 
	double dy = this->y - b->y; 
	double dz = this->z - b->z; 
	return sqrt(dx * dx + dy * dy + dz * dz);
}

double Body::d2x(const Body* b, double a) const {
	double dx = this->x - b->x;
	return a * dx / this->dist(b);
}

double Body::d2y(const Body* b, double a) const {
	double dy = this->y - b->y;
	return a * dy / this->dist(b);
}

double Body::d2z(const Body* b, double a) const {
	double dz = this->z - b->z;
	return a * dz / this->dist(b);
}

// step this one body forward in time
void Body::step_forward(double dt) {
	x += vx * dt;
	y += vy * dt;
	z += vz * dt;	
}
/*
bodies [0] [1] [2] [3] ...
      /     |
    /       \
body0       body1

*/
vector<Body*> bodies;

// step all bodies forward in time
void step_forward(double dt) { 
	for (int i = 0; i < bodies.size(); i++) {
		Body* b = bodies[i];
		for (int j = 0; j < bodies.size(); j++) {
			if (i == j)
				continue;
			Body* other = bodies[j];
			double r = b->dist(other);
			double F = G * b->m * other->m / (r*r);
			double a = F / b->m;
			b->vx -= b->d2x(other, a);
			b->vy -= b->d2y(other, a);
			b->vz -= b->d2z(other, a);
		}
	}
	// step forward after we calculate everyone's velocity
	for (int i = 0; i < bodies.size(); i++) {
		bodies[i]->step_forward(dt);
	}
}

void print(double t) {
	cout << setw(12) << t << '\n';
	for (int i = 0; i < bodies.size(); i++) {
		cout << *bodies[i] << '\n';
	}
	cout << '\n';
}

int main(int argc, char* argv[]) {
	constexpr double YEAR = 365.25 * 24 * 60 * 60;
	constexpr double MONTH = 30 * 24 * 60 * 60;
	const double dt = argc > 1 ? atof(argv[1]) : 100; // 100 second default timestep
	const double END = argc > 2 ? atof(argv[2]) * YEAR : YEAR;
	double print_interval = argc > 3 ? atof(argv[3]) : 1000;
	bodies.push_back(new Body("Sun", 1.989e30, 0,-200,0, 0,0,0));
	bodies.push_back(new Body("Earth", 5.97219e24, 149.59787e9,0,0, 0,29784.8,0));
	bodies.push_back(new Body("Mars", -6.39e23, 228e9,0,0, 0,-24130.8,0));
	bodies.push_back(new Body("Ceres", -9.3839e20, 0, 449e9,0, -16.9e3,0,0));

	cerr << "dt=" << dt << "\tEND=" << (END/YEAR) << " years \tprint=" << print_interval << '\n';
	print_interval *= dt;
	double next_print;
	for (double t = 0; t < END; ) {
		print(t);
		next_print = t + print_interval;
		for (; t < next_print; t += dt) {
			step_forward(dt);
		}
	}
}
