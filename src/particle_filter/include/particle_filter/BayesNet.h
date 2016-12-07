#ifndef BAYESNET_H
#define BAYESNET_H
#include <vector>
#include <array>
#include <cstring>
#include <unordered_set>
#include "definitions.h"
#include "Node.h"

using namespace std;

class BayesNet {
public:
  static const int cdim = 6;
  static const int fulldim = 30;
  int numParticles; // number of particles
  int maxNumParticles;

  // vector<Node *> node;
  FullJoint fullJoint;
  FullJoint fullJointPrev;
  
  Eigen::MatrixXd cov_mat;
  int numFullJoint;
  int numNode;
  BayesNet();
  void addRoot(int numParticles, cspace b_init[2]);
  void createFullJoint(cspace b_Xprior[2]);
  bool updateFullJoint(double cur_M[2][3], double Xstd_ob, double R, int idx);
  void estimateGaussian(cspace &x_mean, cspace &x_est_stat, int idx);
  void getAllParticles(Particles &particles_dest, int idx);
  // vector<cspace> priorSample();
};


#endif // BAYESNET_H