#include <iostream>
#include <cmath>

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
    // without friend:  a.dist(b,c) WRONG
    friend double distsq(const Vec3d& a, const Vec3d& b) {
        double dx = b.x - a.x;
        double dy = b.y - a.y;
        double dz = b.z - a.z;
        return dx*dx+dy*dy+dz*dz;
    }

    friend double dist(const Vec3d& a, const Vec3d& b) {
        return sqrt(distsq(a,b));
    }
};

class Body { 
public:
    const static double G = 5.61e-11; // universal gravitational constant
    double mass;
    Vec3d  pos;
    Vec3d  v;
    Body(double m, const Vec3d& pos, const Vec3d& v)
      : mass(m), pos(pos), v(v) {}
}


class System {
private:
    vector<Body> bodies;
public:
    //possible:    System(const char filename[]);
    System(int numPlanets); // create a random solar system with reasonable orbits
    void stepForward(double dt);
};

void System::stepForward(double dt) {
    for (uint32_t i = 0; i < bodies.size(); i++) {
        Body b = bodies[i]; // temp copy of each body
        for (uint32_t j = 0; j < bodies.size(); j++) {
            double d = dist(b.pos, bodies[j].pos);
            double F = Body::G * b.mass * bodies[j].mass / (d*d);

        }
    }

}



int main() {
    // every unit is in SI (dist=meter)

        // your job is create a set of objects moving "Reasonable" orbits
    Body earth(6.0E+24, Vec3d(150e9,0,0), Vec3d(0, 23e3,0) );

}

