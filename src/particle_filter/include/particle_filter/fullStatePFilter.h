#ifndef FULLSTATEPFILTER_H
#define FULLSTATEPFILTER_H
#include <vector>
#include <array>
#include <cstring>
#include <unordered_set>
#include "definitions.h"
#include "distanceTransformNew.h"

using namespace std;

class fullStatePFilter {
public:
  static const int cdim;
  static const int fulldim;
  int numParticles; // number of particles
  int maxNumParticles;
  double Xstd_ob;
  // vector<Node *> node;
  FullJoint fullJoint;
  FullJoint fullJointPrev;
  Particles holeConfigs;
  
  Eigen::MatrixXd cov_mat;
  int numFullJoint;
  int numNode;
  fullStatePFilter();
  void addRoot(int numParticles, jointCspace b_init[2], double Xstd_ob);
  void createFullJoint(jointCspace b_Xprior[2]);
  bool updateFullJoint(double cur_M[2][3], double Xstd_ob, double R, int idx);
  bool updateFullJoint(double cur_M[2][3], vector<vec4x3> &mesh, distanceTransform *dist_transform, double Xstd_ob, double R, int idx);
  void estimateGaussian(cspace &x_mean, cspace &x_est_stat, int idx);
  void estimateHole(cspace &x_mean, cspace &x_est_stat);
  void getAllParticles(Particles &particles_dest, int idx);
  void getHoleParticles(Particles &particles_dest);
  void buildDistTransform(double cur_M[2][3], vector<vec4x3> &mesh, distanceTransform *dist_transform, int nodeidx);
  void generateHole(jointCspace &joint, int right_datum, int top_datum, int plane, double holeOffset1, double holeOffset2, cspace &hole);
  // vector<cspace> priorSample();
};

void Transform(double measure[2][3], cspace src, double dest[2][3]);
void inverseTransform(double measure[3], cspace src, double dest[3]);
void inverseTransform(double measure[2][3], cspace src, double dest[2][3]);

void Transform(Eigen::Vector3d &src, cspace config, Eigen::Vector3d &dest);
void inverseTransform(Eigen::Vector3d &src, cspace config, Eigen::Vector3d &dest);

void transPointConfig(cspace &baseConfig, cspace &relativeConfig, cspace &absoluteConfig);
void transFrameConfig(cspace baseConfig, cspace relativeConfig, cspace &absoluteConfig);
void invTransFrameConfig(cspace baseConfig, cspace relativeConfig, cspace &absoluteConfig);
void copyParticles(cspace config, fullCspace &fullConfig, int idx);
void copyParticles(cspace config, jointCspace &jointConfig, int idx);
int checkEmptyBin(std::unordered_set<string> *set, cspace config);
void calcDistance(vector<vec4x3> &mesh, cspace trueConfig, cspace meanConfig, double euclDist[2]);
int checkIntersections(vector<vec4x3> &mesh, double voxel_center[3], double dir[3], double check_length, double &dist);
#endif // FULLSTATEPFILTER_H