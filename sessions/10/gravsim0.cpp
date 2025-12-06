#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

constexpr auto G = 5.67E-11; // universal gravitational constant

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
	vector<string> names;
	vector<double> masses; // rest mass
	double x,y,z; // location
	double vx, vy, vz; // velocity
#endif

class Body final {
	string name;
	double m; // rest mass
	double x, y, z; // location
	double vx, vy, vz; // velocity
public:
	Body([[maybe_unused]] const string& name, const double m, const double x, const double y, const double z,
		 const double vx, const double vy, const double vz) : m(m), x(x), y(y), z(z), vx(vx), vy(vy), vz(vz) {}
	~Body();
	friend void step_forward(double dt);
	friend ostream& operator<<(ostream& s, const Body& b) {
		return s << setw(14) << b.name << setw(14) << b.x << setw(14) << b.y << setw(14) << b.vx << setw(14) << b.vy;
	}
	double dist(const Body* b) const;

	// given the acceleration, calculate the component in the x, y, z direction
	double d2x(const Body* b, double a) const;
	double d2y(const Body* b, double a) const;
	double d2z(const Body* b, double a) const;
	void step_forward(double dt);
};

Body::~Body() = default;

double Body::dist(const Body* b) const {
	const auto dx = this->x - b->x;
	const auto dy = this->y - b->y;
	const auto dz = this->z - b->z;
	return sqrt(dx * dx + dy * dy + dz * dz);
}

double Body::d2x(const Body* b, const double a) const { return a * (this->x - b->x) / this->dist(b); }
double Body::d2y(const Body* b, const double a) const { return a * (this->y - b->y) / this->dist(b); }
double Body::d2z(const Body* b, const double a) const { return a * (this->z - b->z) / this->dist(b); }

// step this one body forward in time
void Body::step_forward(const double dt) {
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
void step_forward(const double dt) {
	for (auto i = 0UL; i < bodies.size(); i++) {
		const auto b = bodies[i];
		for (auto j = 0UL; j < bodies.size(); j++) {
			if (i == j)
				continue;
			const auto other = bodies[j];
			const auto r = b->dist(other);
			const auto F = G * b->m * other->m / (r * r);
			const auto a = F / b->m;
			b->vx -= b->d2x(other, a);
			b->vy -= b->d2y(other, a);
			b->vz -= b->d2z(other, a);
		}
	}
	// step forward after we calculate everyone's velocity
	for (const auto& body : bodies)
		body->step_forward(dt);
}

void print(const double t) {
	cout << setw(12) << t << endl;
	for (const auto& body : bodies)
		cout << *body << endl;
	cout << endl;
}

int main(const int argc, char* argv[]) {
	constexpr auto YEAR = 365.25 * 24 * 60 * 60;
	constexpr auto MONTH = 30.0 * 24 * 60 * 60;
	const auto dt = argc > 1 ? strtof(argv[1], nullptr) : 100; // 100 second default timestep
	const auto END = argc > 2 ? strtof(argv[2], nullptr) * YEAR : YEAR;
	auto print_interval = argc > 3 ? strtof(argv[3], nullptr) : 1000;
	bodies.push_back(new Body("Sun", 1.989e30, 0, -200, 0, 0, 0, 0));
	bodies.push_back(new Body("Earth", 5.97219e24, 149.59787e9, 0, 0, 0, 29784.8, 0));
	bodies.push_back(new Body("Mars", -6.39e23, 228e9, 0, 0, 0, -24130.8, 0));
	bodies.push_back(new Body("Ceres", -9.3839e20, 0, 449e9, 0, -16.9e3, 0, 0));

	cerr << "dt=" << dt << "\tEND=" << (END / YEAR) << " years \tprint=" << print_interval << endl;
	print_interval *= dt;
	for (auto t = 0.0; t < END;) {
		print(t);
		for (const auto next_print = t + print_interval; t < next_print; t += dt)
			step_forward(dt);
	}
}
