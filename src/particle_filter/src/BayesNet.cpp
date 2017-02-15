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
// #include "Node.h"
#include "matrix.h"
#include "tribox.h"
#include "raytri.h"
#include "circleEllipse.h"
#include "BayesNet.h"

using namespace std;

const int BayesNet::cdim = 6;
const int BayesNet::fulldim = FULLDIM;
BayesNet::BayesNet()
{
}

void BayesNet::addRoot(int n_particles, jointCspace b_init[2], double Xstdob)
{
  // Node *root = new Node(numParticles, b_init);
  // node.push_back(root);
  // node.push_back(root->child[0]);
  // node.push_back(root->child[1]);
  // node.push_back(root->child[2]);
  // node.push_back(root->child[0]->child[0]);
  // numNode = node.size();
  Xstd_ob = Xstdob;
  numParticles = n_particles;
  maxNumParticles = numParticles;
  fullJoint.resize(numParticles);
  fullJointPrev.resize(numParticles);
  holeConfigs.resize(numParticles);
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

void BayesNet::createFullJoint(jointCspace b_Xprior[2]) {
  
  std::random_device rd;
  std::normal_distribution<double> dist(0, 1);
  cspace tmpConfig;
  for (int i = 0; i < numParticles; i ++) {
    int pt = 0;
    // Root
    for (int j = 0; j < cdim; j++) {
      fullJointPrev[i][j] = b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
      tmpConfig[j] = fullJointPrev[i][j];
    }

    // Front Plane
    cspace relativeConfig, baseConfig, transformedConfig, edgeConfig;
    cspace frontPlaneConfig, rightPlaneConfig, leftPlaneConfig, topPlaneConfig;

    for (int j = 0; j < cdim; j ++) {
      relativeConfig[j] = b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    }
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, frontPlaneConfig);
    //TEMP:
    // if (frontPlaneConfig[5] < 0)  frontPlaneConfig[5] += 2 * Pi;
    copyParticles(frontPlaneConfig, fullJointPrev[i], cdim);

    // // Bottom Edge
    // cspace prior1[2] = {{0,0,0,1.22,0,0},{0,0,0,0.0005,0.0005,0.0005}};
    // for (int j = 0; j < cdim; j++) {
    //   relativeConfig[j] = prior1[0][j] + prior1[1][j] * (dist(rd));
    // }
    // baseConfig = tmpConfig;
    // transPointConfig(baseConfig, relativeConfig, edgeConfig);
    // copyParticles(edgeConfig, fullJointPrev[i], 2 * cdim);

    // // Side Edge
    // cspace prior2[2] = {{0,-0.025,0,0,-0.025,0.23},{0,0,0,0.0005,0.0005,0.0005}};
    // for (int j = 0; j < cdim; j++) {
    //   relativeConfig[j] = prior2[0][j] + prior2[1][j] * (dist(rd));
    // }
    // baseConfig = tmpConfig;
    // transPointConfig(baseConfig, relativeConfig, transformedConfig);
    // copyParticles(transformedConfig, fullJointPrev[i], 3 * cdim);

    // // Top edge
    // double edgeTol = 0.001;
    // double edgeOffSet = 0.23;
    // Eigen::Vector3d pa, pb; 
    // pa << edgeConfig[0], edgeConfig[1], edgeConfig[2];
    // pb << edgeConfig[3], edgeConfig[4], edgeConfig[5];
    // Eigen::Vector3d pa_prime, pb_prime;
    // inverseTransform(pa, frontPlaneConfig, pa_prime);
    // inverseTransform(pb, frontPlaneConfig, pb_prime);
    // double td = dist(rd) * edgeTol;
    // // pa_prime(1) = 0;
    // // pb_prime(1) = 0;
    // Eigen::Vector3d normVec;
    // normVec << (pb_prime(2) - pa_prime(2)), 0, (pa_prime(0) - pb_prime(0));
    // normVec.normalize();
    // normVec *= (edgeOffSet + td);
    // pa_prime(0) += normVec(0);
    // pb_prime(0) += normVec(0);
    // pa_prime(2) += normVec(2);
    // pb_prime(2) += normVec(2);
    // Transform(pa_prime, frontPlaneConfig, pa);
    // Transform(pb_prime, frontPlaneConfig, pb);
    // fullJointPrev[i][24] = pa(0);
    // fullJointPrev[i][25] = pa(1);
    // fullJointPrev[i][26] = pa(2);
    // fullJointPrev[i][27] = pb(0);
    // fullJointPrev[i][28] = pb(1);
    // fullJointPrev[i][29] = pb(2);

    // Top Plane
    for (int j = 0; j < cdim; j ++) {
      relativeConfig[j] = b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    }
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, topPlaneConfig);
    copyParticles(topPlaneConfig, fullJointPrev[i], 2 * cdim);

    // Right Plane
    for (int j = 0; j < cdim; j ++) {
      relativeConfig[j] = b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    }
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, rightPlaneConfig);
    copyParticles(rightPlaneConfig, fullJointPrev[i], 3 * cdim);

    // Left Plane
    for (int j = 0; j < cdim; j ++) {
      relativeConfig[j] = b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    }
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, leftPlaneConfig);
    copyParticles(leftPlaneConfig, fullJointPrev[i], 4 * cdim);

    // // Top Plane
    // relativeConfig[0] = 0 + dist(rd) * 0.001;
    // relativeConfig[1] = -0.063 + dist(rd) * 0.001;
    // relativeConfig[2] = 0.23 + dist(rd) * 0.001;
    // relativeConfig[3] = -1.570796 + dist(rd) * 0.01;
    // relativeConfig[4] = 0 + dist(rd) * 0.01;
    // relativeConfig[5] = 0 + dist(rd) * 0.01;
    // baseConfig = tmpConfig;
    // transFrameConfig(baseConfig, relativeConfig, topPlaneConfig);
    // copyParticles(topPlaneConfig, fullJointPrev[i], 7 * cdim);


    // Hole 
    generateHole(fullJointPrev[i], 3, 2, 1, 0.1, 0.1, holeConfigs[i]);

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
  Eigen::MatrixXd H_cov = coeff * cov_mat;


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

      count += 1;
      if (D <= signed_dist_check) {
        count2 ++;
        for (int j = 0; j < fulldim; j++) {
          fullJoint[i][j] = tempFullState[j];
        }
        if (checkEmptyBin(&bins, tempState) == 1) {
          num_bins++;
          numParticles = min2(maxNumParticles, max2(((num_bins - 1) * 2), N_MIN));
          // }
        }
        i += 1;
      }
    }
    cout << "End updating Plane!" << endl;
  } else { // Edge
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
          
      count += 1;
      if (D <= signed_dist_check) {
        count2 ++;
        for (int j = 0; j < fulldim; j++) {
          fullJoint[i][j] = tempFullState[j];
        }
        if (checkEmptyBin(&bins, tempState) == 1) {
          num_bins++;
          numParticles = min2(maxNumParticles, max2(((num_bins - 1) * 2), N_MIN));
        }
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

  Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)fullJoint.data(), fulldim, numParticles);
  Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
  cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));

  cout << "End updating!" << endl;
  return iffar;
}

void BayesNet::buildDistTransform(double cur_M[2][3], vector<vec4x3> &mesh, distanceTransform *dist_transform, int nodeidx) {
  int num_Mean = numParticles;
  std::vector<std::array<double,3>> measure_workspace;
  measure_workspace.resize(num_Mean);
  std::random_device rd;
  std::uniform_real_distribution<double> distribution(0, numParticles);
  double var_measure[3] = { 0, 0, 0 };
  cspace meanConfig = { 0, 0, 0, 0, 0, 0 };
  double mean_inv_M[3];
  double distTransSize;
  cspace b_X;
  int startIdx = nodeidx * cdim;
  for (int t = 0; t < num_Mean; t++) {
    int index = int(floor(distribution(rd)));
    for (int m = 0; m < cdim; m++) {
      meanConfig[m] += fullJointPrev[index][m + startIdx];
    }
    for (int m = 0; m < cdim; m++) {
      b_X[m] = fullJointPrev[index][m + startIdx];
    }
    inverseTransform(cur_M[0], b_X, measure_workspace[t].data());
  }
  for (int m = 0; m < cdim; m++) {
    meanConfig[m] /= num_Mean;
  }
  // inverse-transform using sampled configuration
  inverseTransform(cur_M[0], meanConfig, mean_inv_M);
  for (int t = 0; t < num_Mean; t++) {
    var_measure[0] += SQ(measure_workspace[t][0] - mean_inv_M[0]);
    var_measure[1] += SQ(measure_workspace[t][1] - mean_inv_M[1]);
    var_measure[2] += SQ(measure_workspace[t][2] - mean_inv_M[2]);
  }
  var_measure[0] /= num_Mean;
  var_measure[1] /= num_Mean;
  var_measure[2] /= num_Mean;
  distTransSize = max2(2 * max3(sqrt(var_measure[0]), sqrt(var_measure[1]), sqrt(var_measure[2])), 20 * Xstd_ob);
  // distTransSize = 100 * 0.0005;
  cout << "Touch Std: " << sqrt(var_measure[0]) << "  " << sqrt(var_measure[1]) << "  " << sqrt(var_measure[2]) << endl;
  double world_range[3][2];
  cout << "Current Inv_touch: " << mean_inv_M[0] << "    " << mean_inv_M[1] << "    " << mean_inv_M[2] << endl;
  for (int t = 0; t < 3; t++) {
    world_range[t][0] = mean_inv_M[t] - distTransSize;
    world_range[t][1] = mean_inv_M[t] + distTransSize;
    /*cout << world_range[t][0] << " to " << world_range[t][1] << endl;*/
  }
    
  dist_transform->voxelizeSTL(mesh, world_range);
  dist_transform->build();
  
}

bool BayesNet::updateFullJoint(double cur_M[2][3], vector<vec4x3> &mesh, distanceTransform *dist_transform, double Xstd_ob, double R, int nodeidx) {
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
  Eigen::MatrixXd H_cov = coeff * cov_mat;


  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(H_cov);
  Eigen::MatrixXd rot = eigenSolver.eigenvectors(); 
  Eigen::VectorXd scl = eigenSolver.eigenvalues();
  for (int j = 0; j < fulldim; j++) {
    scl(j, 0) = sqrt(max2(scl(j, 0),0));
  }
  Eigen::VectorXd samples(fulldim, 1);
  Eigen::VectorXd rot_sample(fulldim, 1);
  buildDistTransform(cur_M, mesh, dist_transform, nodeidx);
  cout << "Start updating datum: " + nodeidx << endl;
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
    if (cur_inv_M[0][0] >= dist_transform->world_range[0][1] || 
      cur_inv_M[0][0] <= dist_transform->world_range[0][0] ||
      cur_inv_M[0][1] >= dist_transform->world_range[1][1] || 
      cur_inv_M[0][1] <= dist_transform->world_range[1][0] ||
      cur_inv_M[0][2] >= dist_transform->world_range[2][1] || 
      cur_inv_M[0][2] <= dist_transform->world_range[2][0]) {
      continue;
    }
      
    int xind = int(floor((cur_inv_M[0][0] - dist_transform->world_range[0][0]) / 
               dist_transform->voxel_size));
    int yind = int(floor((cur_inv_M[0][1] - dist_transform->world_range[1][0]) / 
               dist_transform->voxel_size));
    int zind = int(floor((cur_inv_M[0][2] - dist_transform->world_range[2][0]) / 
               dist_transform->voxel_size));

    // D = abs(cur_inv_M[0][1] - R);
    D = (*dist_transform->dist_transform)[xind][yind][zind];

    count += 1;
    if (D <= unsigned_dist_check) {
      count2 ++;
      if (checkIntersections(mesh, cur_inv_M[0], cur_inv_M[1], ARM_LENGTH, D)) {
        count_bar ++;
        if (count_bar > 1000)
        break;
        continue;
      }
      count_bar = 0;
      D -= R;
    }
    else
      continue;
    if (D >= -signed_dist_check && D <= signed_dist_check) {

      for (int j = 0; j < fulldim; j++) {
        fullJoint[i][j] = tempFullState[j];
      }
      generateHole(fullJoint[i], 3, 2, 1, 0.1, 0.1, holeConfigs[i]);
      if (checkEmptyBin(&bins, tempState) == 1) {
        num_bins++;
        numParticles = min2(maxNumParticles, max2(((num_bins - 1) * 2), N_MIN));
        // }
      }
      i += 1;
    }
  }
  cout << "Number of total iterations: " << count << endl;
  cout << "Number of iterations after unsigned_dist_check: " << count2 << endl;
  cout << "Number of iterations before safepoint check: " << count3 << endl;
  cout << "Number of occupied bins: " << num_bins << endl;
  cout << "Number of particles: " << numParticles << endl;
  
  fullJointPrev = fullJoint;

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
}
void BayesNet::getHoleParticles(Particles &particles_dest) {
  particles_dest.resize(numParticles);
  particles_dest = holeConfigs;
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

void BayesNet::generateHole(jointCspace &joint, int right_datum, int top_datum, int plane, double holeOffset1, double holeOffset2, cspace &hole) {
  Eigen::Vector3d pa1, pb1, pa2, pb2, ta, tb;
  int datum1Start = right_datum * cdim;
  int datum2Start = top_datum * cdim;
  int planeStart = plane * cdim;

  cspace baseConfig1 = {joint[datum1Start], joint[datum1Start + 1], joint[datum1Start + 2], joint[datum1Start + 3], joint[datum1Start + 4], joint[datum1Start + 5]};
  cspace baseConfig2 = {joint[datum2Start], joint[datum2Start + 1], joint[datum2Start + 2], joint[datum2Start + 3], joint[datum2Start + 4], joint[datum2Start + 5]};

  ta << 0, -0.025, 0;
  tb << 0, -0.025, 0.23;
  Transform(ta, baseConfig1, pa1);
  Transform(tb, baseConfig1, pb1);
  ta << 0, -0.025, 0.23;
  tb << 1.2192, -0.025, 0.23;
  Transform(ta, baseConfig2, pa2);
  Transform(tb, baseConfig2, pb2);

  cspace planeConfig;
  for (int i = 0; i < cdim; i ++) {
    planeConfig[i] = joint[planeStart + i];
  }

  Eigen::Vector3d pa1_prime, pb1_prime, pa2_prime, pb2_prime;
  
  inverseTransform(pa1, planeConfig, pa1_prime);
  inverseTransform(pb1, planeConfig, pb1_prime);
  inverseTransform(pa2, planeConfig, pa2_prime);
  inverseTransform(pb2, planeConfig, pb2_prime);
  Eigen::Vector3d normVec;
  normVec << (pb1_prime(2) - pa1_prime(2)), 0, (pa1_prime(0) - pb1_prime(0));
  normVec.normalize();
  normVec *= (holeOffset1);
  pa1_prime(0) += normVec(0);
  pb1_prime(0) += normVec(0);
  pa1_prime(1) = -0.025;
  pb1_prime(1) = -0.025;
  pa1_prime(2) += normVec(2);
  pb1_prime(2) += normVec(2);

  normVec << (pb2_prime(2) - pa2_prime(2)), 0, (pa2_prime(0) - pb2_prime(0));
  normVec.normalize();
  normVec *= (holeOffset2);
  pa2_prime(0) += normVec(0);
  pb2_prime(0) += normVec(0);
  pa2_prime(1) = -0.025;
  pb2_prime(1) = -0.025;
  pa2_prime(2) += normVec(2);
  pb2_prime(2) += normVec(2);

  Eigen::Matrix2d divisor, dividend; 
  divisor << pa1_prime(0) - pb1_prime(0), pa1_prime(2) - pb1_prime(2),
             pa2_prime(0) - pb2_prime(0), pa2_prime(2) - pb2_prime(2);
  dividend << pa1_prime(0)*pb1_prime(2) - pa1_prime(2)*pb1_prime(0), pa1_prime(0) - pb1_prime(0),
              pa2_prime(0)*pb2_prime(2) - pa2_prime(2)*pb2_prime(0), pa2_prime(0) - pb2_prime(0);
  Eigen::Vector3d pi_prime, pi, dir_prime, origin_prime, dir, origin;
  pi_prime(0) = dividend.determinant() / divisor.determinant();
  dividend << pa1_prime(0)*pb1_prime(2) - pa1_prime(2)*pb1_prime(0), pa1_prime(2) - pb1_prime(2),
              pa2_prime(0)*pb2_prime(2) - pa2_prime(2)*pb2_prime(0), pa2_prime(2) - pb2_prime(2);
  pi_prime(1) = -0.025;
  pi_prime(2) = dividend.determinant() / divisor.determinant();
  dir_prime << 0, 1, 0; origin_prime << 0, 0, 0;
  Transform(pi_prime, planeConfig, pi);
  Transform(dir_prime, planeConfig, dir);
  Transform(origin_prime, planeConfig, origin);
  dir -= origin;
  hole[0] = pi(0) - holeOffset1;
  hole[1] = pi(1) - ta(1);
  hole[2] = pi(2) - ta(2) + holeOffset1;
  // hole[3] = dir(0);
  // hole[4] = dir(1);
  // hole[5] = dir(2);
  // hole[3] = asin(dir[2] / sqrt(dir[1] * dir[1] + dir[2] * dir[2]));
  // hole[4] = asin(-dir[2] / sqrt(dir[0] * dir[0] + dir[2] * dir[2]));
  // hole[5] = asin(dir[2] / sqrt(dir[0] * dir[0] + dir[2] * dir[2]));
  hole[3] = planeConfig[3];
  hole[4] = planeConfig[4];
  hole[5] = planeConfig[5];
  // std::cout << hole[3] << " " << hole[4] << " " << hole[5] << std::endl;
}


void BayesNet::estimateHole(cspace &x_mean, cspace &x_est_stat) {
  cout << "Estimated Mean: ";
  for (int k = 0; k < cdim; k++) {
    x_mean[k] = 0;
    for (int j = 0; j < numParticles; j++) {
      x_mean[k] += holeConfigs[j][k];
    }
    x_mean[k] /= numParticles;
    cout << x_mean[k] << "  ";
  }
  cout << endl;
  cout << "Estimated Std: ";
  for (int k = 0; k < cdim; k++) {
    x_est_stat[k] = 0;
    for (int j = 0; j < numParticles; j++) {
      x_est_stat[k] += SQ(holeConfigs[j][k] - x_mean[k]);
    }
    x_est_stat[k] = sqrt(x_est_stat[k] / numParticles);
    cout << x_est_stat[k] << "  ";
  }
  cout << endl;

}













void transFrameConfig(cspace baseConfig, cspace relativeConfig, cspace &absoluteConfig) {
  Eigen::Matrix4d baseTrans, relativeTrans, absoluteTrans;
  Eigen::Matrix3d rotationC, rotationB, rotationA;
  rotationC << cos(baseConfig[5]), -sin(baseConfig[5]), 0,
             sin(baseConfig[5]), cos(baseConfig[5]), 0,
             0, 0, 1;
  rotationB << cos(baseConfig[4]), 0 , sin(baseConfig[4]),
             0, 1, 0,
             -sin(baseConfig[4]), 0, cos(baseConfig[4]);
  rotationA << 1, 0, 0 ,
             0, cos(baseConfig[3]), -sin(baseConfig[3]),
             0, sin(baseConfig[3]), cos(baseConfig[3]);
  Eigen::Matrix3d rotation = rotationC * rotationB * rotationA;
  baseTrans << rotation(0, 0), rotation(0, 1), rotation(0, 2), baseConfig[0],
               rotation(1, 0), rotation(1, 1), rotation(1, 2), baseConfig[1],
               rotation(2, 0), rotation(2, 1), rotation(2, 2), baseConfig[2],
               0,              0,              0,              1;
  rotationC << cos(relativeConfig[5]), -sin(relativeConfig[5]), 0,
             sin(relativeConfig[5]), cos(relativeConfig[5]), 0,
             0, 0, 1;
  rotationB << cos(relativeConfig[4]), 0 , sin(relativeConfig[4]),
             0, 1, 0,
             -sin(relativeConfig[4]), 0, cos(relativeConfig[4]);
  rotationA << 1, 0, 0 ,
             0, cos(relativeConfig[3]), -sin(relativeConfig[3]),
             0, sin(relativeConfig[3]), cos(relativeConfig[3]);
  rotation = rotationC * rotationB * rotationA;
  relativeTrans << rotation(0, 0), rotation(0, 1), rotation(0, 2), relativeConfig[0],
                   rotation(1, 0), rotation(1, 1), rotation(1, 2), relativeConfig[1],
                   rotation(2, 0), rotation(2, 1), rotation(2, 2), relativeConfig[2],
                   0,              0,              0,              1;
  absoluteTrans = baseTrans * relativeTrans;
  absoluteConfig[0] = absoluteTrans(0,3);
  absoluteConfig[1] = absoluteTrans(1,3);
  absoluteConfig[2] = absoluteTrans(2,3);
  absoluteConfig[3] = atan2(absoluteTrans(2,1), absoluteTrans(2,2));
  absoluteConfig[4] = atan2(-absoluteTrans(2,0), sqrt(SQ(absoluteTrans(2,1)) + SQ(absoluteTrans(2,2))));
  absoluteConfig[5] = atan2(absoluteTrans(1,0), absoluteTrans(0,0));
  // cout << absoluteTrans << endl;
}

void invTransFrameConfig(cspace baseConfig, cspace relativeConfig, cspace &absoluteConfig) {
  Eigen::Matrix4d baseTrans, relativeTrans, absoluteTrans;
  Eigen::Matrix3d rotationC, rotationB, rotationA;
  rotationC << cos(baseConfig[5]), -sin(baseConfig[5]), 0,
             sin(baseConfig[5]), cos(baseConfig[5]), 0,
             0, 0, 1;
  rotationB << cos(baseConfig[4]), 0 , sin(baseConfig[4]),
             0, 1, 0,
             -sin(baseConfig[4]), 0, cos(baseConfig[4]);
  rotationA << 1, 0, 0 ,
             0, cos(baseConfig[3]), -sin(baseConfig[3]),
             0, sin(baseConfig[3]), cos(baseConfig[3]);
  Eigen::Matrix3d rotation = rotationC * rotationB * rotationA;
  baseTrans << rotation(0, 0), rotation(0, 1), rotation(0, 2), baseConfig[0],
               rotation(1, 0), rotation(1, 1), rotation(1, 2), baseConfig[1],
               rotation(2, 0), rotation(2, 1), rotation(2, 2), baseConfig[2],
               0,              0,              0,              1;
  baseTrans = baseTrans.inverse().eval();
  rotationC << cos(relativeConfig[5]), -sin(relativeConfig[5]), 0,
             sin(relativeConfig[5]), cos(relativeConfig[5]), 0,
             0, 0, 1;
  rotationB << cos(relativeConfig[4]), 0 , sin(relativeConfig[4]),
             0, 1, 0,
             -sin(relativeConfig[4]), 0, cos(relativeConfig[4]);
  rotationA << 1, 0, 0 ,
             0, cos(relativeConfig[3]), -sin(relativeConfig[3]),
             0, sin(relativeConfig[3]), cos(relativeConfig[3]);
  rotation = rotationC * rotationB * rotationA;
  relativeTrans << rotation(0, 0), rotation(0, 1), rotation(0, 2), relativeConfig[0],
                   rotation(1, 0), rotation(1, 1), rotation(1, 2), relativeConfig[1],
                   rotation(2, 0), rotation(2, 1), rotation(2, 2), relativeConfig[2],
                   0,              0,              0,              1;
  absoluteTrans = baseTrans * relativeTrans;
  absoluteConfig[0] = absoluteTrans(0,3);
  absoluteConfig[1] = absoluteTrans(1,3);
  absoluteConfig[2] = absoluteTrans(2,3);
  absoluteConfig[3] = atan2(absoluteTrans(2,1), absoluteTrans(2,2));
  absoluteConfig[4] = atan2(-absoluteTrans(2,0), sqrt(SQ(absoluteTrans(2,1)) + SQ(absoluteTrans(2,2))));
  absoluteConfig[5] = atan2(absoluteTrans(1,0), absoluteTrans(0,0));
  // cout << absoluteTrans << endl;
}


void transPointConfig(cspace &baseConfig, cspace &relativeConfig, cspace &absoluteConfig) {
  Eigen::Matrix4d baseTrans, relativeTrans, absoluteTrans;
  Eigen::Matrix3d rotationC, rotationB, rotationA;
  rotationC << cos(baseConfig[5]), -sin(baseConfig[5]), 0,
             sin(baseConfig[5]), cos(baseConfig[5]), 0,
             0, 0, 1;
  rotationB << cos(baseConfig[4]), 0 , sin(baseConfig[4]),
             0, 1, 0,
             -sin(baseConfig[4]), 0, cos(baseConfig[4]);
  rotationA << 1, 0, 0 ,
             0, cos(baseConfig[3]), -sin(baseConfig[3]),
             0, sin(baseConfig[3]), cos(baseConfig[3]);
  Eigen::Matrix3d rotation = rotationC * rotationB * rotationA;
  baseTrans << rotation(0, 0), rotation(0, 1), rotation(0, 2), baseConfig[0],
               rotation(1, 0), rotation(1, 1), rotation(1, 2), baseConfig[1],
               rotation(2, 0), rotation(2, 1), rotation(2, 2), baseConfig[2],
               0,              0,              0,              1;
  Eigen::Vector4d endPoint1, endPoint2;
  endPoint1 << relativeConfig[0], relativeConfig[1], relativeConfig[2], 1;
  endPoint2 << relativeConfig[3], relativeConfig[4], relativeConfig[5], 1;
  endPoint1 = baseTrans * endPoint1;
  endPoint2 = baseTrans * endPoint2;
  absoluteConfig[0] = endPoint1(0);
  absoluteConfig[1] = endPoint1(1);
  absoluteConfig[2] = endPoint1(2);
  absoluteConfig[3] = endPoint2(0);
  absoluteConfig[4] = endPoint2(1);
  absoluteConfig[5] = endPoint2(2);
}

void copyParticles(cspace config, fullCspace &fullConfig, int idx) {
  for (int i = 0; i < BayesNet::cdim; i ++) {
    fullConfig[idx + i] = config[i];
  }
}


void copyParticles(cspace config, jointCspace &jointConfig, int idx) {
  for (int i = 0; i < BayesNet::cdim; i ++) {
    jointConfig[idx + i] = config[i];
  }
}

/*
 * Transform the touch point from particle frame
 */
void Transform(Eigen::Vector3d &src, cspace config, Eigen::Vector3d &dest)
{
    Eigen::Matrix3d rotationC;
    rotationC << cos(config[5]), -sin(config[5]), 0,
               sin(config[5]), cos(config[5]), 0,
               0, 0, 1;
    Eigen::Matrix3d rotationB;
    rotationB << cos(config[4]), 0 , sin(config[4]),
               0, 1, 0,
               -sin(config[4]), 0, cos(config[4]);
    Eigen::Matrix3d rotationA;
    rotationA << 1, 0, 0 ,
               0, cos(config[3]), -sin(config[3]),
               0, sin(config[3]), cos(config[3]);
    Eigen::Vector3d transitionV(config[0], config[1], config[2]);
    dest = rotationC * rotationB * rotationA * src + transitionV;
}
/*
 * Transform the touch point from particle frame
 */
void Transform(double measure[2][3], cspace src, double dest[2][3])
{
  double rotation[3][3];
  double tempM[3];
  rotationMatrix(src, rotation);
  multiplyM(rotation, measure[0], tempM);
  multiplyM(rotation, measure[1], dest[1]);
  double transition[3] = { src[0], src[1], src[2] };
  addM(tempM, transition, dest[0]);
}
/*
 * Inverse transform the touch point to particle frame using sampled configuration
 */
void inverseTransform(double measure[3], cspace src, double dest[3])
{
  double rotation[3][3];
  double invRot[3][3];
  double tempM[3];
  rotationMatrix(src, rotation);
  inverseMatrix(rotation, invRot);
  double transition[3] = { src[0], src[1], src[2] };
  subtractM(measure, transition, tempM);
  multiplyM(invRot, tempM, dest);
}
void inverseTransform(double measure[2][3], cspace src, double dest[2][3])
{
  double rotation[3][3];
  double invRot[3][3];
  double tempM[3];
  rotationMatrix(src, rotation);
  inverseMatrix(rotation, invRot);
  double transition[3] = { src[0], src[1], src[2] };
  subtractM(measure[0], transition, tempM);
  multiplyM(invRot, tempM, dest[0]);
  multiplyM(invRot, measure[1], dest[1]);
}
void inverseTransform(Eigen::Vector3d &src, cspace config, Eigen::Vector3d &dest)
{
    Eigen::Matrix3d rotationC;
    rotationC << cos(config[5]), -sin(config[5]), 0,
               sin(config[5]), cos(config[5]), 0,
               0, 0, 1;
    Eigen::Matrix3d rotationB;
    rotationB << cos(config[4]), 0 , sin(config[4]),
               0, 1, 0,
               -sin(config[4]), 0, cos(config[4]);
    Eigen::Matrix3d rotationA;
    rotationA << 1, 0, 0 ,
               0, cos(config[3]), -sin(config[3]),
               0, sin(config[3]), cos(config[3]);
    Eigen::Vector3d transitionV(config[0], config[1], config[2]);
    Eigen::Matrix3d rotationM = rotationC * rotationB * rotationA;
    dest = rotationM.inverse() * (src - transitionV);
}

void calcDistance(vector<vec4x3> &mesh, cspace trueConfig, cspace meanConfig, double euclDist[2])
{
    int num_mesh = int(mesh.size());
    cout << "Num_Mesh " << num_mesh << endl;
    euclDist[0] = 0;
    euclDist[1] = 10000000;
    double dist = 0;
    Eigen::Vector3d meshPoint;
    Eigen::Vector3d transMeanPoint;
    Eigen::Vector3d transTruePoint;
    for (int i = 0; i < num_mesh; i++) {
        meshPoint << mesh[i][1][0], mesh[i][1][1], mesh[i][1][2];
        Transform(meshPoint, meanConfig, transMeanPoint);
        Transform(meshPoint, trueConfig, transTruePoint);
        dist = (transMeanPoint - transTruePoint).norm();
        if (dist > euclDist[0]) {
            euclDist[0] = dist;
        }
        if (dist < euclDist[1]) {
            euclDist[1] = dist;
        }
        meshPoint << mesh[i][2][0], mesh[i][2][1], mesh[i][2][2];
        Transform(meshPoint, meanConfig, transMeanPoint);
        Transform(meshPoint, trueConfig, transTruePoint);
        dist = (transMeanPoint - transTruePoint).norm();
        if (dist > euclDist[0]) {
            euclDist[0] = dist;
        }
        if (dist < euclDist[1]) {
            euclDist[1] = dist;
        }
        meshPoint << mesh[i][3][0], mesh[i][3][1], mesh[i][3][2];
        Transform(meshPoint, meanConfig, transMeanPoint);
        Transform(meshPoint, trueConfig, transTruePoint);
        dist = (transMeanPoint - transTruePoint).norm();
        if (dist > euclDist[0]) {
            euclDist[0] = dist;
        }
        if (dist < euclDist[1]) {
            euclDist[1] = dist;
        }
    }
}

/* 
 * Check if the configuration falls into an empty bin
 * Input: set: Set to store non-empty bins
 *        config: Sampled particle
 * Output: 1 if empty
 *         0 if not, and then add the bin to set
 */
int checkEmptyBin(std::unordered_set<string> *set, cspace config)
{
  string s = "";
  for (int i = 0; i < BayesNet::cdim; i++) {
    s += floor(config[i] / DISPLACE_INTERVAL);
    s += ":";
  }
  if (set->find(s) == set->end()) {
    set->insert(s);
    return 1;
  }
  return 0;
}

int checkIntersections(vector<vec4x3> &mesh, double voxel_center[3], double dir[3], double check_length, double &dist)
{
  int countIntersections = 0;
  int countIntRod = 0;
  int num_mesh = int(mesh.size());
  double vert0[3], vert1[3], vert2[3];
  double *t = new double;
  double *u = new double;
  double *v = new double;
  double tMax = 0;
  double ray_dir[3] = {-dir[0], -dir[1], -dir[2]};
  Eigen::Vector3d normal_dir;
  Eigen::Vector3d ray_length;
  double inside_length;
  std::unordered_set<double> hashset;
  //std::unordered_map<double, int> hashmap;
  for (int i = 0; i < num_mesh; i++) {
    vert0[0] = mesh[i][1][0];
    vert0[1] = mesh[i][1][1];
    vert0[2] = mesh[i][1][2];
    vert1[0] = mesh[i][2][0];
    vert1[1] = mesh[i][2][1];
    vert1[2] = mesh[i][2][2];
    vert2[0] = mesh[i][3][0];
    vert2[1] = mesh[i][3][1];
    vert2[2] = mesh[i][3][2];
    if (intersect_triangle(voxel_center, ray_dir, vert0, vert1, vert2, t, u, v) == 1) {
      if (hashset.find(*t) == hashset.end()) {
        if (*t < check_length && *t > tMax) {
          countIntRod++;
          tMax = *t;
          normal_dir << mesh[i][0][0], mesh[i][0][1], mesh[i][0][2];
        }
        else if (*t < check_length)
          countIntRod++;
        hashset.insert(*t);
        countIntersections++;
      }
    }
  }
  delete t, u, v;
  if (countIntersections % 2 == 0) {
    if (tMax > 0)
      return 1;
    return 0;
  }
  else {
  dist = -dist;
  if (tMax > 0 && countIntRod % 2 == 1) {
    ray_length << tMax * dir[0], tMax * dir[1], tMax * dir[2];
    double inter_dist = normal_dir.dot(ray_length);
    if (inter_dist >= dist - EPSILON && inter_dist <= dist + EPSILON)
      return 0;
  }
  return 1;
  }
}