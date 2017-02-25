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
#include "fullStatePFilter.h"

using namespace std;

const int fullStatePFilter::cdim = 6;
const int fullStatePFilter::fulldim = FULLDIM;
fullStatePFilter::fullStatePFilter()
{
}

void fullStatePFilter::addRoot(int n_particles, jointCspace b_init[2], double Xstdob)
{
  Xstd_ob = Xstdob;
  numParticles = n_particles;
  maxNumParticles = numParticles;
  fullJoint.resize(numParticles);
  fullJointPrev.resize(numParticles);
  holeConfigs.resize(numParticles);
  createFullJoint(b_init);
}

void fullStatePFilter::createFullJoint(jointCspace b_Xprior[2]) {
  
  std::random_device rd;
  std::normal_distribution<double> dist(0, 1);
  cspace tmpConfig;
  for (int j = 0; j < cdim; j++) {
      tmpConfig[j] = 0;
  }
  for (int i = 0; i < numParticles; i ++) {
    int pt = 0;
    // // Root
    // for (int j = 0; j < cdim; j++) {
    //   fullJointPrev[i][j] = b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    //   tmpConfig[j] = fullJointPrev[i][j];
    // }

    cspace topConfig, frontConfig, relativeConfig, baseConfig, transformedConfig, edgeConfig;
    cspace frontPlaneConfig, rightPlaneConfig, leftPlaneConfig, topPlaneConfig;

    // Front Plane
    for (int j = 0; j < cdim; j ++) {
      relativeConfig[j] = b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    }
    frontConfig = relativeConfig;
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, frontPlaneConfig);
    copyParticles(frontPlaneConfig, fullJointPrev[i], 0);

    // Top Plane
    for (int j = 0; j < cdim; j ++) {
      relativeConfig[j] = relativeConfig[j] + b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    }
    topConfig = relativeConfig;
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, topPlaneConfig);
    copyParticles(topPlaneConfig, fullJointPrev[i], cdim);

    // Right Plane
    for (int j = 0; j < cdim; j ++) {
      relativeConfig[j] = relativeConfig[j] + b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    }
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, rightPlaneConfig);
    copyParticles(rightPlaneConfig, fullJointPrev[i], 2 * cdim);

    // Left Plane
    for (int j = 0; j < cdim; j ++) {
      relativeConfig[j] = relativeConfig[j] + b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    }
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, leftPlaneConfig);
    copyParticles(leftPlaneConfig, fullJointPrev[i], 3 * cdim);

    // Bottom Plane
    for (int j = 0; j < cdim; j ++) {
      relativeConfig[j] = topConfig[j] + b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    }
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, leftPlaneConfig);
    copyParticles(leftPlaneConfig, fullJointPrev[i], 4 * cdim);

    // J1 Section
    for (int j = 0; j < cdim; j ++) {
      relativeConfig[j] = frontConfig[j] + b_Xprior[0][pt] + b_Xprior[1][pt ++] * dist(rd);
    }
    baseConfig = tmpConfig;
    transFrameConfig(baseConfig, relativeConfig, leftPlaneConfig);
    copyParticles(leftPlaneConfig, fullJointPrev[i], 5 * cdim);

    // Hole 
    generateHole(fullJointPrev[i], 2, 1, 0, 1.1192, 0.1, holeConfigs[i]);

  }
  fullJoint = fullJointPrev;
  Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)fullJoint.data(), fulldim, numParticles);
  Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
  cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
}


void fullStatePFilter::buildDistTransform(double cur_M[2][3], vector<vec4x3> &mesh, distanceTransform *dist_transform, int nodeidx) {
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

// /*
//  * Update particles (Build distance transform and sampling)
//  * Input: cur_M: current observation
//  *        mesh: object mesh arrays
//  *        dist_transform: distance transform class instance
//  *        R: radius of the touch probe
//  *        Xstd_ob: observation error
//  *        nodeidx: idx of datum measured
//  * output: return whether previous estimate is bad (not used here)
//  */

bool fullStatePFilter::updateFullJoint(double cur_M[2][3], vector<vec4x3> &mesh, distanceTransform *dist_transform, double Xstd_ob, double R, int nodeidx) {
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

  Eigen::Vector3d touch_dir;
  int num_bins = 0;
  int count_bar = 0;
  double coeff = pow(4.0 / ((fulldim + 2.0) * numParticles), 2.0/(fulldim + 4.0)) /1.2155/1.2155/1.5;
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
      generateHole(fullJoint[i], 2, 1, 0, 1.1192, 0.1, holeConfigs[i]);
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

void fullStatePFilter::getAllParticles(Particles &particles_dest, int idx)
{
  particles_dest.resize(numParticles);
  for (int j = 0; j < numParticles; j++) {
    for (int k = 0; k < cdim; k++) {
      particles_dest[j][k] = fullJoint[j][k + idx * cdim];
    }
  }
}
void fullStatePFilter::getHoleParticles(Particles &particles_dest) {
  particles_dest.resize(numParticles);
  particles_dest = holeConfigs;
}

void fullStatePFilter::estimateGaussian(cspace &x_mean, cspace &x_est_stat, int idx) {
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

void fullStatePFilter::generateHole(jointCspace &joint, int right_datum, int top_datum, int plane, double holeOffset1, double holeOffset2, cspace &hole) {
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
  hole[0] = pi(0);
  hole[1] = pi(1);
  hole[2] = pi(2);
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


void fullStatePFilter::estimateHole(cspace &x_mean, cspace &x_est_stat) {
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
  for (int i = 0; i < fullStatePFilter::cdim; i ++) {
    fullConfig[idx + i] = config[i];
  }
}


void copyParticles(cspace config, jointCspace &jointConfig, int idx) {
  for (int i = 0; i < fullStatePFilter::cdim; i ++) {
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
  for (int i = 0; i < fullStatePFilter::cdim; i++) {
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