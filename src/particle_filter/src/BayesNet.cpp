#include <vector>
#include <array>
#include <cstring>
#include <unordered_set>
#include <iostream>
#include <random>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>

#include "definitions.h"
#include "Node.h"
#include "BayesNet.h"

using namespace std;


BayesNet::BayesNet()
{
}

void BayesNet::addRoot(int n_particles, cspace b_init[2])
{
  // Node *root = new Node(numParticles, b_init);
  // node.push_back(root);
  // node.push_back(root->child[0]);
  // node.push_back(root->child[1]);
  // node.push_back(root->child[2]);
  // node.push_back(root->child[0]->child[0]);
  // numNode = node.size();
  numParticles = n_particles;
  maxNumParticles = numParticles;
  fullJoint.resize(numParticles);
  fullJointPrev.resize(numParticles);
  createFullJoint(b_init);

  // for (int i = 0; i < numNode; i ++) {
  //   Node *cur = node[i];
  //   for (int j = 0; j < numParticles; j ++) {
  //     for (int k = 0; k < 6) {
  //       fullJoint[j][i * 6 + k] = cur->particles[j][k];
  //     }
  //   }
  // }
}

void BayesNet::createFullJoint(cspace b_Xprior[2]) {
  
  std::random_device rd;
  std::normal_distribution<double> dist(0, 1);
  cspace tmpConfig;
  for (int i = 0; i < numParticles; i ++) {
    // Root
    for (int j = 0; j < cdim; j++) {
      fullJointPrev[i][j] = b_Xprior[0][j] + b_Xprior[1][j] * (dist(rd));
      tmpConfig[j] = fullJointPrev[i][j];
    }

    // Front Plane
    cspace relativeConfig, baseConfig, transformedConfig, edgeConfig;
    cspace frontPlaneConfig, sidePlaneConfig, otherSidePlaneConfig;
    relativeConfig[0] = 1.22;
    relativeConfig[1] = -0.025;
    relativeConfig[2] = 0;
    relativeConfig[3] = 0;
    relativeConfig[4] = 0;
    relativeConfig[5] = Pi;
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, frontPlaneConfig);
    //TEMP:
    if (frontPlaneConfig[5] < 0)  frontPlaneConfig[5] += 2 * Pi;
    copyParticles(frontPlaneConfig, fullJointPrev[i], cdim);

    // Bottom Edge
    cspace prior1[2] = {{0,0,0,1.22,0,0},{0,0,0,0.0005,0.0005,0.0005}};
    for (int j = 0; j < cdim; j++) {
      relativeConfig[j] = prior1[0][j] + prior1[1][j] * (dist(rd));
    }
    baseConfig = tmpConfig;
    transPointConfig(baseConfig, relativeConfig, edgeConfig);
    copyParticles(edgeConfig, fullJointPrev[i], 2 * cdim);

    // Side Edge
    cspace prior2[2] = {{0,-0.025,0,0,-0.025,0.23},{0,0,0,0.0005,0.0005,0.0005}};
    for (int j = 0; j < cdim; j++) {
      relativeConfig[j] = prior2[0][j] + prior2[1][j] * (dist(rd));
    }
    baseConfig = tmpConfig;
    transPointConfig(baseConfig, relativeConfig, transformedConfig);
    copyParticles(transformedConfig, fullJointPrev[i], 3 * cdim);

    // Top edge
    double edgeTol = 0.001;
    double edgeOffSet = 0.23;
    Eigen::Vector3d pa, pb; 
    pa << edgeConfig[0], edgeConfig[1], edgeConfig[2];
    pb << edgeConfig[3], edgeConfig[4], edgeConfig[5];
    Eigen::Vector3d pa_prime, pb_prime;
    inverseTransform(pa, frontPlaneConfig, pa_prime);
    inverseTransform(pb, frontPlaneConfig, pb_prime);
    double td = dist(rd) * edgeTol;
    // pa_prime(1) = 0;
    // pb_prime(1) = 0;
    Eigen::Vector3d normVec;
    normVec << (pb_prime(2) - pa_prime(2)), 0, (pa_prime(0) - pb_prime(0));
    normVec.normalize();
    normVec *= (edgeOffSet + td);
    pa_prime(0) += normVec(0);
    pb_prime(0) += normVec(0);
    pa_prime(2) += normVec(2);
    pb_prime(2) += normVec(2);
    Transform(pa_prime, frontPlaneConfig, pa);
    Transform(pb_prime, frontPlaneConfig, pb);
    fullJointPrev[i][24] = pa(0);
    fullJointPrev[i][25] = pa(1);
    fullJointPrev[i][26] = pa(2);
    fullJointPrev[i][27] = pb(0);
    fullJointPrev[i][28] = pb(1);
    fullJointPrev[i][29] = pb(2);

    // Side Plane
    relativeConfig[0] = 0;
    relativeConfig[1] = 0;
    relativeConfig[2] = 0;
    relativeConfig[3] = 0;
    relativeConfig[4] = 0;
    relativeConfig[5] = -Pi / 2.0;
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, sidePlaneConfig);
    copyParticles(sidePlaneConfig, fullJointPrev[i], 5 * cdim);

    // Other Side Plane
    relativeConfig[0] = 1.24 + dist(rd) * 0.03;
    // relativeConfig[0] = 1.2192;
    relativeConfig[1] = 0;
    relativeConfig[2] = 0;
    relativeConfig[3] = 0;
    relativeConfig[4] = 0;
    relativeConfig[5] = Pi / 2.0;
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, otherSidePlaneConfig);
    copyParticles(otherSidePlaneConfig, fullJointPrev[i], 6 * cdim);
    // Hole 


  }
  fullJoint = fullJointPrev;
  Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)fullJoint.data(), fulldim, numParticles);
  Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
  cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
}
// vector<cspace> BayesNet::priorSample() 
// {
//   vector<cspace> sample;
//   sample.resize(numNode);

// }

bool BayesNet::updateFullJoint(double cur_M[2][3], double Xstd_ob, double R, int nodeidx) {
  cout << "Start updating!" << endl;
  std::unordered_set<string> bins;
  std::random_device rd;
  std::normal_distribution<double> dist(0, 1);
  std::uniform_real_distribution<double> distribution(0, numParticles);
  int i = 0;
  int count = 0;
  int count2 = 0;
  int count3 = 0;
  bool iffar = false;
  FullJoint b_X = fullJointPrev;
  int idx = 0;
  jointCspace tempFullState;
  cspace tempState;
  double D;
  double cur_inv_M[2][3];
  
  double unsigned_dist_check = R + 1 * Xstd_ob;
  double signed_dist_check = 1 * Xstd_ob;

  //Eigen::Vector3d gradient;
  Eigen::Vector3d touch_dir;
  int num_bins = 0;
  int count_bar = 0;
  double coeff = pow(4.0 / ((fulldim + 2.0) * numParticles), 2.0/(fulldim + 4.0)) /1.2155/1.2155;
  // cout << "Coeff: " << coeff << endl;
  Eigen::MatrixXd H_cov = coeff * cov_mat;
  // cout << "full_cov_mat: " << full_cov_mat << endl;
  // cout << "cov_mat: " << cov_mat << endl;
  // cout << "H_cov: " << H_cov << endl;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(H_cov);
  Eigen::MatrixXd rot = eigenSolver.eigenvectors(); 
  Eigen::VectorXd scl = eigenSolver.eigenvalues();
  for (int j = 0; j < fulldim; j++) {
    scl(j, 0) = sqrt(max2(scl(j, 0),0));
  }
  Eigen::VectorXd samples(fulldim, 1);
  Eigen::VectorXd rot_sample(fulldim, 1);
  if (nodeidx == 1 || nodeidx == 5 || nodeidx == 6) { // Plane
    cout << "Start updating Plane!" << endl;
    while (i < numParticles && i < maxNumParticles) {
      idx = int(floor(distribution(rd)));

      for (int j = 0; j < fulldim; j++) {
        samples(j, 0) = scl(j, 0) * dist(rd);
      }
      rot_sample = rot*samples;
      for (int j = 0; j < fulldim; j++) {
        /* TODO: use quaternions instead of euler angles */
        tempFullState[j] = b_X[idx][j] + rot_sample(j, 0);
      }
      for (int j = 0; j < cdim; j++) {
        /* TODO: use quaternions instead of euler angles */
        tempState[j] = tempFullState[j + nodeidx * cdim];
      }
      inverseTransform(cur_M, tempState, cur_inv_M);
      touch_dir << cur_inv_M[1][0], cur_inv_M[1][1], cur_inv_M[1][2];
      D = abs(cur_inv_M[0][1] - R);
      // cout << "D: " << D << endl;
      
      // if (xind >= (dist_transform->num_voxels[0] - 1) || yind >= (dist_transform->num_voxels[1] - 1) || zind >= (dist_transform->num_voxels[2] - 1))
      //  continue;
          
      count += 1;
      // if (sqrt(count) == floor(sqrt(count))) cout << "DDDD " << D << endl;
      if (D <= signed_dist_check) {
        // if (sqrt(count) == floor(sqrt(count))) cout << "D " << D << endl;
        count2 ++;
        for (int j = 0; j < fulldim; j++) {
          fullJoint[i][j] = tempFullState[j];
        }
        if (checkEmptyBin(&bins, tempState) == 1) {
          num_bins++;
          // if (i >= N_MIN) {
          //int numBins = bins.size();
          numParticles = min2(maxNumParticles, max2(((num_bins - 1) * 2), N_MIN));
          // }
        }
        //double d = testResult(mesh, particles[i], cur_M, R);
        //if (d > 0.01)
        //  cout << cur_inv_M[0][0] << "  " << cur_inv_M[0][1] << "  " << cur_inv_M[0][2] << "   " << d << "   " << D << //"   " << gradient << "   " << gradient.dot(touch_dir) << 
        //       "   " << dist_adjacent[0] << "   " << dist_adjacent[1] << "   " << dist_adjacent[2] << "   " << particles[i][2] << endl;
        i += 1;
      }
    }
    cout << "End updating Plane!" << endl;
  } else { // Edge
    // for (int i = 0; i < numParticles; i ++) {
    //   // for (int j = 0; j < cdim; j ++) {
    //   //   cout << particles[i][j] << " ,";
    //   // }
    //   cout << particles[i][0] << " ,";
      
    // }
    // cout << endl;
    // for (int i = 0; i < numParticles; i ++) {
    //   // for (int j = 0; j < 18; j ++) {
    //   //   cout << fullParticles[i][j] << " ,";
    //   // }
    //   cout << fullParticles[i][0] << " ,";
    // }
    //       cout << endl;

    while (i < numParticles && i < maxNumParticles) {
      idx = int(floor(distribution(rd)));

      for (int j = 0; j < fulldim; j++) {
        samples(j, 0) = scl(j, 0) * dist(rd);
      }
      rot_sample = rot*samples;
      for (int j = 0; j < fulldim; j++) {
        /* TODO: use quaternions instead of euler angles */
        tempFullState[j] = b_X[idx][j] + rot_sample(j, 0);
      }

      for (int j = 0; j < cdim; j++) {
        /* TODO: use quaternions instead of euler angles */
        tempState[j] = tempFullState[j + nodeidx * cdim];
      }
      Eigen::Vector3d x1, x2, x0, tmp1, tmp2;
      x1 << tempState[0], tempState[1], tempState[2];
      x2 << tempState[3], tempState[4], tempState[5];
      x0 << cur_M[0][0], cur_M[0][1], cur_M[0][2];
      tmp1 = x1 - x0;
      tmp2 = x2 - x1;
      D = (tmp1.squaredNorm() * tmp2.squaredNorm() - pow(tmp1.dot(tmp2),2)) / tmp2.squaredNorm();
      D = abs(sqrt(D)- R);
      // cout << "Cur distance: " << D << endl;
      // cout << "D: " << D << endl;
      
      // if (xind >= (dist_transform->num_voxels[0] - 1) || yind >= (dist_transform->num_voxels[1] - 1) || zind >= (dist_transform->num_voxels[2] - 1))
      //  continue;
          
      count += 1;
      // if (sqrt(count) == floor(sqrt(count))) cout << "DDDD " << D << endl;
      if (D <= signed_dist_check) {
        // if (sqrt(count) == floor(sqrt(count))) cout << "D " << D << endl;
        count2 ++;
        for (int j = 0; j < fulldim; j++) {
          fullJoint[i][j] = tempFullState[j];
        }
        if (checkEmptyBin(&bins, tempState) == 1) {
          num_bins++;
          // if (i >= N_MIN) {
          //int numBins = bins.size();
          numParticles = min2(maxNumParticles, max2(((num_bins - 1) * 2), N_MIN));
          // }
        }
        //double d = testResult(mesh, particles[i], cur_M, R);
        //if (d > 0.01)
        //  cout << cur_inv_M[0][0] << "  " << cur_inv_M[0][1] << "  " << cur_inv_M[0][2] << "   " << d << "   " << D << //"   " << gradient << "   " << gradient.dot(touch_dir) << 
        //       "   " << dist_adjacent[0] << "   " << dist_adjacent[1] << "   " << dist_adjacent[2] << "   " << particles[i][2] << endl;
        i += 1;
      }
    }
    cout << "End updating Edge!" << endl;    
  }
  cout << "Number of total iterations: " << count << endl;
  cout << "Number of iterations after unsigned_dist_check: " << count2 << endl;
  cout << "Number of iterations before safepoint check: " << count3 << endl;
  cout << "Number of occupied bins: " << num_bins << endl;
  cout << "Number of particles: " << numParticles << endl;
  
  fullJointPrev = fullJoint;

  // double* tmpParticles = new double[numParticles * fulldim];
  // for(int i = 0; i < numParticles; ++i) {
  //   for (int j = 0; j < fulldim; j ++) {
  //     tmpParticles[i * fulldim + j] = fullParticlesPrev[i][j];
  //   }
  // }

  Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)fullJoint.data(), fulldim, numParticles);
  Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
  cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));

  cout << "End updating!" << endl;
  return iffar;
}

void BayesNet::getAllParticles(Particles &particles_dest, int idx)
{
  particles_dest.resize(numParticles);
  for (int j = 0; j < numParticles; j++) {
    for (int k = 0; k < cdim; k++) {
      particles_dest[j][k] = fullJoint[j][k + idx * cdim];
    }
  }
  cout << "finish get Particles" << endl;
}

void BayesNet::estimateGaussian(cspace &x_mean, cspace &x_est_stat, int idx) {
  cout << "Estimated Mean: ";
  for (int k = 0; k < cdim; k++) {
    x_mean[k] = 0;
    for (int j = 0; j < numParticles; j++) {
      x_mean[k] += fullJoint[j][k + idx * cdim];
    }
    x_mean[k] /= numParticles;
    cout << x_mean[k] << "  ";
  }
  cout << endl;
  cout << "Estimated Std: ";
  for (int k = 0; k < cdim; k++) {
    x_est_stat[k] = 0;
    for (int j = 0; j < numParticles; j++) {
      x_est_stat[k] += SQ(fullJoint[j][k + idx * cdim] - x_mean[k]);
    }
    x_est_stat[k] = sqrt(x_est_stat[k] / numParticles);
    cout << x_est_stat[k] << "  ";
  }
  cout << endl;

}