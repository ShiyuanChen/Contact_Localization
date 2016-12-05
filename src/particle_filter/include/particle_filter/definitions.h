#ifndef DEFINITIONS_H
#define DEFINITIONS_H
#include <vector>
#include <array>

using namespace std;

#define CDIM 6

typedef array<array<float, 3>, 4> vec4x3;
typedef std::array<double,CDIM> cspace; // configuration space of the particles
typedef std::vector<double> fullCspace;
typedef std::vector<cspace> Particles;
typedef std::vector<fullCspace> FullParticles;


#endif // DEFINITIONS_H