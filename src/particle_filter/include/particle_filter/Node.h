#ifndef NODE_H
#define NODE_H
#include <vector>
#include <array>
#include <cstring>
#include <unordered_set>
#include <Eigen/Dense>
#include "definitions.h"
#include "distanceTransformNew.h"

using namespace std;
class Parent;
class fullStatePFilter;
class particleFilter;
class Node
{
  friend class particleFilter;
  friend class Parent;
  friend class fullStatePFilter;
 public:
  static const int cdim = 6;
  int numParticles; // number of particles
  int maxNumParticles;
  int fulldim;

  Node (int n_particles, cspace b_init[2]);
  Node (int n_particles, std::vector<Parent *> &p, int type);
  Node (int n_particles, std::vector<Parent *> &p, cspace b_init[2]);
  // void addObservation (double obs[2][3], vector<vec4x3> &mesh, distanceTransform *dist_transform, bool miss = false);
  void estimateGaussian(cspace &x_mean, cspace &x_est_stat);
  void getAllParticles(Particles &particles_dest);
  void getPriorParticles(Particles &particles_dest, int idx);
 protected:
  // Parameters of Node
  int type; // 0: root; 1: plane; 2. edge; 3. hole
  double length;
  bool numUpdates;
  // double Xstd_ob; // observation measurement error
  // double R; // probe radius

  // internal variables
  cspace b_Xprior[2]; // Initial distribution (mean and variance)
  //cspace b_Xpre[2];   // Previous (estimated) distribution (mean and variance)
  std::vector<Node*> child;
  std::vector<Parent*> parent;
  Particles particles;
  Particles particlesPrev;
  FullParticles fullParticles;
  FullParticles fullParticlesPrev;

  Eigen::MatrixXd cov_mat;
  Eigen::MatrixXd full_cov_mat;

  // Local functions
  void createParticles(cspace b_Xprior[2], int n_particles, int isRoot);
  void createParticles();
  void addDatum(double dist, double tol);
  // bool update(double cur_M[2][3], double Xstd_ob, double R);
  void sampleConfig(cspace &config); // sample a config from the particles uniformly.
  void propagate();
  void resampleParticles(Particles &rootParticles, Particles &rootParticlesPrev, int n, double *W);
  void sampleParticles();

  // void calcWeight
};

class Parent
{
  friend class particleFilter;
  friend class Node;
public:
  // Parent(Node *p, double x, double y, double z, double tx, double ty, double tz) {
  //   node = p;
  //   type = p->type;
  //   offset[0] = x;
  //   offset[1] = y;
  //   offset[2] = z;
  //   tol[0] = tx;
  //   tol[1] = ty;
  //   tol[2] = tz;
  // }
  Parent(Node *p, double offset, double tol):offset(offset),tol(tol) {
    node = p;
    type = p->type;
  }
protected:
  Node *node;
  int type;
  // double tol[3];
  // double offset[3];
  double offset;
  double tol;

};

void Transform(double measure[2][3], cspace src, double dest[2][3]);
void inverseTransform(double measure[3], cspace src, double dest[3]);
void inverseTransform(double measure[2][3], cspace src, double dest[2][3]);

void Transform(Eigen::Vector3d &src, cspace config, Eigen::Vector3d &dest);
void inverseTransform(Eigen::Vector3d &src, cspace config, Eigen::Vector3d &dest);

void transPointConfig(cspace baseConfig, cspace relativeConfig, cspace &absoluteConfig);
void transFrameConfig(cspace baseConfig, cspace relativeConfig, cspace &absoluteConfig);
void invTransFrameConfig(cspace baseConfig, cspace relativeConfig, cspace &absoluteConfig);
void copyParticles(cspace config, fullCspace &fullConfig, int idx);
void copyParticles(cspace config, jointCspace &jointConfig, int idx);
int checkEmptyBin(std::unordered_set<string> *set, cspace config);
void calcDistance(vector<vec4x3> &mesh, cspace trueConfig, cspace meanConfig, double euclDist[2]);
int checkIntersections(vector<vec4x3> &mesh, double voxel_center[3], double dir[3], double check_length, double &dist);
#endif // NODE_H

