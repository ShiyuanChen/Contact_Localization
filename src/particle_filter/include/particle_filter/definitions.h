#ifndef DEFINITIONS_H
#define DEFINITIONS_H
#include <vector>
#include <array>

using namespace std;

#define CDIM 6
#define FULLDIM 36

typedef array<array<float, 3>, 4> vec4x3;
typedef std::array<double,CDIM> cspace; // configuration space of the particles
typedef std::vector<double> fullCspace;
typedef std::vector<cspace> Particles;
typedef std::vector<fullCspace> FullParticles;
typedef std::vector<array<double, FULLDIM>> FullJoint;
typedef std::array<double,FULLDIM> jointCspace;

#define Pi          3.141592653589793238462643383279502884L

#define SQ(x) ((x)*(x))
#define max3(a,b,c) ((a>b?a:b)>c?(a>b?a:b):c)
#define max2(a,b) (a>b?a:b)
#define min3(a,b,c) ((a<b?a:b)<c?(a<b?a:b):c)
#define min2(a,b) (a<b?a:b)

#define COMBINE_RAYCASTING
#define ADAPTIVE_NUMBER
#define ADAPTIVE_BANDWIDTH

#define EPSILON 0.0001
#define ARM_LENGTH 0.8
#define N_MIN 50
#define DISPLACE_INTERVAL 0.015
#define SAMPLE_RATE 0.50
#define MAX_ITERATION 100000
#define COV_MULTIPLIER 5.0
#define MIN_STD 1.0e-7
#define BEAM_RADIUS 0.002
#define BEAM_STEPSIZE 0.001
#define NUM_POLY_ITERATIONS 20

#endif // DEFINITIONS_H