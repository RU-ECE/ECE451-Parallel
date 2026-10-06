#include <iostream>
#include <cmath>
#include <random>
#include <vector>

/*

    Gravity Simulator
    Given: n=10^6 bodies
    for each body : 7 variables * 8 bytes/double * n = 56MB
        mass
        x,y,z
        vx,vy,vz

        ax,ay,az


    F(a,b) = G m_1 M_2 / (dist(a,b)^2)

    O(n^2)

    a(body) = F(body,bi), for all bi
*/

class Vec3d {
public:
    double x,y,z;
    Vec3d(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    // without friend:  a.dist(b,c) WRONG
    friend double distsq(const Vec3d& a, const Vec3d& b) {
        double dx = b.x - a.x;
        double dy = b.y - a.y;
        double dz = b.z - a.z;
        return dx*dx+dy*dy+dz*dz;
    }
    friend Vec3d operator+(const Vec3d& a, const Vec3d& b) {
        return Vec3d(a.x + b.x, a.y + b.y, a.z + b.z);
    }
    friend Vec3d operator-(const Vec3d& a, const Vec3d& b) {
        return Vec3d(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    friend Vec3d operator*(const Vec3d& a, double s) {
        return Vec3d(a.x * s, a.y * s, a.z * s);
    }
    friend Vec3d operator*(double s, const Vec3d& a) {
        return Vec3d(a.x * s, a.y * s, a.z * s);
    }
    friend Vec3d operator/(const Vec3d& a, double s) {
        return Vec3d(a.x / s, a.y / s, a.z / s);
    }
    Vec3d& operator+=(const Vec3d& b) {
        x += b.x; y += b.y; z += b.z;
        return *this;
    }
    friend double dist(const Vec3d& a, const Vec3d& b) {
        return sqrt(distsq(a,b));
    }
    friend std::ostream& operator<<(std::ostream& s, const Vec3d& a) {
        return s << a.x << ',' << a.y << ',' << a.z;
    }
};

class Body { 
public:
    static constexpr double G = 6.674e-11; // universal gravitational constant
    double mass;
    Vec3d  pos;
    Vec3d  v;
    Vec3d  a;
    Body(double m, const Vec3d& pos, const Vec3d& v)
      : mass(m), pos(pos), v(v), a(0,0,0) {}
};


class System {
private:
    std::vector<Body> bodies;
    void add_sun(double mass);
    void add_circular(double mass, double r, double angle);
    void add_population(std::mt19937& rng, int count,
                        double mlo, double mhi, double rmin, double rmax);
public:
    //possible:    System(const char filename[]);
    System(int n); // sun, 5-20 planets, remaining n are asteroids
    void stepForward(double dt);
    void print() const;
};

// v = sqrt(GM/r) keeps a circular orbit around a fixed central mass
static double circular_speed(double central_mass, double r) {
    return std::sqrt(Body::G * central_mass / r);
}

static double random_unit(std::mt19937& rng) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

// log-uniform so masses span orders of magnitude, not just the top of the range
static double random_log(std::mt19937& rng, double lo, double hi) {
    return lo * std::pow(hi / lo, random_unit(rng));
}

void System::add_sun(double mass) {
    bodies.emplace_back(mass, Vec3d(0, 0, 0), Vec3d(0, 0, 0));
}

// position on the circle, velocity tangent, same sense for every body (ccw)
void System::add_circular(double mass, double r, double angle) {
    double v = circular_speed(bodies[0].mass, r);
    bodies.emplace_back(
        mass,
        Vec3d(r * std::cos(angle), r * std::sin(angle), 0),
        Vec3d(-v * std::sin(angle), v * std::cos(angle), 0));
}

void System::add_population(std::mt19937& rng, int count,
                            double mlo, double mhi, double rmin, double rmax) {
    const double two_pi = 2.0 * std::acos(-1.0);
    for (int i = 0; i < count; i++) {
        double mass = random_log(rng, mlo, mhi);
        double r = rmin + random_unit(rng) * (rmax - rmin);
        double angle = random_unit(rng) * two_pi;
        add_circular(mass, r, angle);
    }
}

// sun, then 5-20 planets, then the rest of n as asteroids
System::System(int n) {
    bodies.reserve(n + 1);
    const double AU = 1.496e11;
    add_sun(1.9885e30);
    std::mt19937 rng(0); // fixed seed so a run is repeatable
    int nplanets = std::uniform_int_distribution<int>(5, 20)(rng);
    if (nplanets > n)
        nplanets = n;
    // Mercury..Jupiter scale, spread from inside Mercury out past Neptune
    add_population(rng, nplanets, 3.3e23, 1.9e27, 0.4 * AU, 30.0 * AU);
    // small bodies packed into an asteroid belt
    add_population(rng, n - nplanets, 1.0e15, 1.0e20, 2.1 * AU, 3.3 * AU);
}

void System::stepForward(double dt) {
    for (uint32_t i = 0; i < bodies.size(); i++) {
        bodies[i].a = Vec3d(0,0,0);
        for (uint32_t j = 0; j < bodies.size(); j++) {
            if (i == j) continue;
            // a_i = G m_j / d^2 along the unit vector (pos_j - pos_i)/d
            double d = dist(bodies[i].pos, bodies[j].pos);
            double s = Body::G * bodies[j].mass / (d*d*d);
            bodies[i].a += (bodies[j].pos - bodies[i].pos) * s;
        }
        // this is not correct. Can you think why?
        // bodies[i].v += bodies[i].a * dt;
        // bodies[i].pos += bodies[i].v * dt;
    }
    /*
      this is correct. Why do we have to do it after?
      this is not a good algorithm. The error is O(dt)
      and the error is cumulative.

      In this homework, you do not have to implement better
      algorithms (but you may). The first simplest algorithm
      is RKF45. This is very complex, and very clever.
      Runge-Kutta-Fehlberg is a family of algorithms.
      RKF45 computes a 4th order and a 5th order solution
      that happen to use some of the same calculations to save time.
      This allows estimating the error by comparing 4th and 5th order
      solutions.      
    */
    for (uint32_t i = 0; i < bodies.size(); i++) {  
        bodies[i].v += bodies[i].a * dt;
        bodies[i].pos += bodies[i].v * dt;
    }
}

void System::print() const {
    for (uint32_t i = 0; i < bodies.size(); i++)
        std::cout << i << '\t' << bodies[i].mass << '\t'
                  << bodies[i].pos << '\t' << bodies[i].v << '\n';
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_bodies> [dt] [num_steps] [num_output]" << std::endl;
        return 1;
    }
    const int num_bodies = std::stoi(argv[1]);
    const double dt = argc > 2 ? std::stod(argv[2]) : 10.0; // 10 second timestep
    const int num_steps = argc > 3 ? std::stoi(argv[3]) : 100;
    const int num_output = argc > 4 ? std::stoi(argv[4]) : 10;
    System system(num_bodies);
    for (int time = 0; time < num_steps; time++) {
        system.stepForward(dt);
        if (time % num_output == 0) {
            system.print();
        }
    }
    return 0;
}