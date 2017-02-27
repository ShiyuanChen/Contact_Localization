#include <string.h>
#include <iostream>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <chrono>
#include <cmath> 
#include "definitions.h"
#include "tribox.h"
#include "raytri.h"
#include "circleEllipse.h"
#include "distanceTransformNew.h"
#include "particleFilter.h"
#include "matrix.h"
#include "stlParser.h"
#include "fullStatePFilter.h"

using namespace std;

int total_time = 0;
int converge_count = 0;
double TRUE_STATE[6] = {0.3, 0.3, 0.3, 0.5, 0.7, 0.5};

//vector<vec4x3> importSTL(string filename);

const int particleFilter::cdim = 6;
/*
 * particleFilter class Construction
 * Input: n_particles: number of particles
 *        b_init[2]: prior belief
 *        Xstd_ob: observation error
 *        Xstd_tran: standard deviation of gaussian kernel when sampling
 *        Xstd_scatter: scatter param before sampling the mean of dist_trans
 *        R: radius of the touch probe
 * output: none
 */
particleFilter::particleFilter(int n_particles, jointCspace b_init[2],
							   double Xstd_ob, double R)
  : numParticles(n_particles), maxNumParticles(n_particles), 
    Xstd_ob(Xstd_ob), R(R)
{
  cspace trueConfig = {1.1192, -0.025, 0.13, 0, 0, 0};
  b_Xprior[0] = b_init[0];
  b_Xprior[1] = b_init[1];
  fullStateFilter.addRoot(numParticles, b_Xprior, Xstd_ob);
  cspace particles_mean, tmp2;
  estimateGaussian(particles_mean, tmp2);
  cout << "Estimate diff: ";
  double est_diff = sqrt(SQ(particles_mean[0] - trueConfig[0]) + SQ(particles_mean[1] - trueConfig[1]) + SQ(particles_mean[2] - trueConfig[2])
                       + SQ(particles_mean[3] - trueConfig[3]) + SQ(particles_mean[5] - trueConfig[5]));
  cout << est_diff << endl;
  double est_diff_trans = sqrt(SQ(particles_mean[0] - trueConfig[0]) + SQ(particles_mean[1] - trueConfig[1]) + SQ(particles_mean[2] - trueConfig[2]));
  double est_diff_rot = sqrt(SQ(particles_mean[3] - trueConfig[3]) + SQ(particles_mean[5] - trueConfig[5]));
  ofstream myfile;
  myfile.open("/home/shiyuan/Documents/ros_marsarm/diff.csv", ios::out|ios::app);
  myfile << est_diff << ",";
  myfile.close();
  myfile.open("/home/shiyuan/Documents/ros_marsarm/diff_trans.csv", ios::out|ios::app);
  myfile << est_diff_trans << ",";
  myfile.close();
  myfile.open("/home/shiyuan/Documents/ros_marsarm/diff_rot.csv", ios::out|ios::app);
  myfile << est_diff_rot << ",";
  myfile.close();
  // particles.resize(numParticles);
  // particlesPrev.resize(numParticles);

  // createParticles(particlesPrev, b_Xprior, numParticles);
  // particles = particlesPrev;

// #ifdef ADAPTIVE_BANDWIDTH
//   Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)particlesPrev.data(), cdim, numParticles);
//   Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
//   cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
//   cout << cov_mat << endl;
// #endif
  //W = new double[numParticles];
}


void particleFilter::getHoleParticles(Particles &particles_dest) {
  fullStateFilter.getHoleParticles(particles_dest);
}

void particleFilter::getAllParticles(Particles &particles_dest)
{
  // fullStateFilter.node[0]->child[0]->getAllParticles(particles_dest);
  fullStateFilter.getAllParticles(particles_dest, 0);
  // root->child[1]->child[0]->getPriorParticles(particles_dest, cdim);
}

void particleFilter::getAllParticles(Particles &particles_dest, int idx)
{
  // fullStateFilter.node[0]->child[0]->getAllParticles(particles_dest);
  fullStateFilter.getAllParticles(particles_dest, idx);
  // root->child[1]->child[0]->getPriorParticles(particles_dest, cdim);
}

// /*
//  * Create initial particles at start
//  * Input: particles
//  *        b_Xprior: prior belief
//  *        n_partcles: number of particles
//  * output: none
//  */
// void particleFilter::createParticles(Particles &particles_dest, cspace b_Xprior[2],
// 									 int n_particles)
// {
//   random_device rd;
//   normal_distribution<double> dist(0, 1);
//   int cdim = sizeof(cspace) / sizeof(double);
//   for (int i = 0; i < n_particles; i++) {
// 		for (int j = 0; j < cdim; j++) {
// 		  particles_dest[i][j] = b_Xprior[0][j] + b_Xprior[1][j] * (dist(rd));
// 		}
//   }
// }

/*
 * Add new observation and call updateParticles() to update the particles
 * Input: obs: observation
 *        mesh: object mesh arrays
 *        dist_transform: distance transform class instance
 *        miss: if it is a miss touch
 * output: none
 */
void particleFilter::addObservation(double obs[2][3], vector<vec4x3> &mesh, distanceTransform *dist_transform, bool miss, int datum)
{
	// cspace trueConfig = {0.3, 0.3, 0.3, 0.5, 0.7, 0.5};
	cspace trueConfig = {1.1192, -0.025, 0.13, 0, 0, 0};
  cout << "Xstd_Ob: " << Xstd_ob << endl;
  auto timer_begin = std::chrono::high_resolution_clock::now();
  std::random_device generator;

  // bool iffar = root->updateParticles(obs, mesh, dist_transform, Xstd_ob, R, miss);
  // bool iffar = fullStateFilter.node[0]->child[1]->child[0]->update(obs, Xstd_ob, R);
  // bool iffar = fullStateFilter.updateFullJoint(obs, Xstd_ob, R, datum);

  bool iffar = fullStateFilter.updateFullJoint(obs, mesh, dist_transform, Xstd_ob, R, datum);

  auto timer_end = std::chrono::high_resolution_clock::now();
  auto timer_dur = timer_end - timer_begin;
  numParticles = fullStateFilter.numParticles;

  cspace particles_mean, tmp2;
  estimateGaussian(particles_mean, tmp2);
  cout << "Estimate diff: ";
  double est_diff = sqrt(SQ(particles_mean[0] - trueConfig[0]) + SQ(particles_mean[1] - trueConfig[1]) + SQ(particles_mean[2] - trueConfig[2])
                       + SQ(particles_mean[3] - trueConfig[3]) + SQ(particles_mean[5] - trueConfig[5]));
  cout << est_diff << endl;
  if (est_diff >= 0.0015) {
    converge_count ++;
  }
  double est_diff_trans = sqrt(SQ(particles_mean[0] - trueConfig[0]) + SQ(particles_mean[1] - trueConfig[1]) + SQ(particles_mean[2] - trueConfig[2]));
  double est_diff_rot = sqrt(SQ(particles_mean[3] - trueConfig[3]) + SQ(particles_mean[5] - trueConfig[5]));

  cout << "Converge count: " << converge_count << endl;
  cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timer_dur).count() << endl;
  total_time += std::chrono::duration_cast<std::chrono::milliseconds>(timer_dur).count();
  cout << "Total time: " << total_time << endl;
  cout << "Average time: " << total_time / 20.0 << endl;
  double euclideanDist[2];
  calcDistance(mesh, trueConfig, particles_mean, euclideanDist);
  cout << "Maximum workspace distance: " << euclideanDist[0] << endl;
  cout << "Minimum workspace distance: " << euclideanDist[1] << endl << endl;

  ofstream myfile;
  myfile.open("/home/shiyuan/Documents/ros_marsarm/diff.csv", ios::out|ios::app);
  myfile << est_diff << ",";
  myfile.close();
  myfile.open("/home/shiyuan/Documents/ros_marsarm/time.csv", ios::out|ios::app);
  myfile << std::chrono::duration_cast<std::chrono::milliseconds>(timer_dur).count() << ",";
  myfile.close();
  myfile.open("/home/shiyuan/Documents/ros_marsarm/diff_trans.csv", ios::out|ios::app);
  myfile << est_diff_trans << ",";
  myfile.close();
  myfile.open("/home/shiyuan/Documents/ros_marsarm/diff_rot.csv", ios::out|ios::app);
  myfile << est_diff_rot << ",";
  myfile.close();
  // myfile.open("/home/shiyuan/Documents/ros_marsarm/workspace_max.csv", ios::out|ios::app);
  // myfile << euclideanDist[0] << ",";
  // myfile.close();
  // myfile.open("/home/shiyuan/Documents/ros_marsarm/workspace_min.csv", ios::out|ios::app);
  // myfile << euclideanDist[1] << ",";
  // myfile.close();
}

void particleFilter::estimateGaussian(cspace &x_mean, cspace &x_est_stat) {
  fullStateFilter.estimateHole(x_mean, x_est_stat);
  cspace trueConfig = {1.1192, -0.025, 0.13, 0, 0, 0};
  ofstream myfile;
  myfile.open("/home/shiyuan/Documents/ros_marsarm/rate.csv", ios::out|ios::app);
  myfile << computeHoleError(fullStateFilter.holeConfigs, trueConfig, 0.01, 0.035, 0.8, 20) << ",";
  myfile.close();
}


double particleFilter::computeHoleError(Particles &holeConfigs, cspace &trueConfig, double circle_radius, double hole_depth,
                                      double fit_ratio, int num_poly_iterations) {
  // double ray_start[3] = {1.1192, 0, 0.13};
  // double ray_end[3] = {1.1192, -0.15, 0.13};
  double ray_start[3] = {0, 0.05, 0};
  double ray_end[3] = {0, -0.1, 0};
  double circle_center[2][3] = {{0, 0, 0}, {0, 1, 0}};
  // double trans_start[3];
  // double trans_end[3];
  double temp_center[2][3];
  double trans_center[2][3];
  // Transform(ray_start, particles_mean, trans_start);
  // Transform(ray_end, particles_mean, trans_end);
  
  double rate = 0;
  double fitted_radius = fit_ratio * circle_radius;
  // double cumError = 0;
  circleEllipse cEllipse(num_poly_iterations);
  for (int ii = 0; ii < numParticles; ii ++) {

    Transform(circle_center, holeConfigs[ii], temp_center);
    inverseTransform(temp_center, trueConfig, trans_center);
    Eigen::Vector3d proj_vec;
    proj_vec << 0, 1, 0;
    Eigen::Vector3d center_vec;
    center_vec << trans_center[1][0], trans_center[1][1], trans_center[1][2];
    //center_vec /= center_vec.norm();
    double cos_view_angle = abs(center_vec.dot(proj_vec));
    double sin_view_angle = sqrt(1 - cos_view_angle * cos_view_angle);
    double depth_offset = hole_depth * sin_view_angle;
    double minor_length = cos_view_angle * circle_radius;
    Eigen::Vector2d minor_vec;
    minor_vec << -center_vec[0], center_vec[2];
    //cout << "minor _vec " << minor_vec << endl;
    double minor_angle = atan2(-center_vec[0], center_vec[2]);
    //cout << "minor _angle  " << minor_angle << endl;
    Eigen::Matrix2d rot;
    rot << cos(minor_angle), -sin(minor_angle),
           sin(minor_angle), cos(minor_angle);

    Eigen::Vector2d projected_ray, vert_dir, hori_dir;
    projected_ray << ray_start[0] + trans_center[0][0], ray_start[2] - trans_center[0][2];
    projected_ray = rot * projected_ray;

    if (cEllipse.circleInEllipse(fitted_radius, projected_ray(0), projected_ray(1), circle_radius, minor_length) &&
      cEllipse.circleInEllipse(fitted_radius, projected_ray(0), projected_ray(1) + depth_offset, circle_radius, minor_length)) {
      rate ++;
      // cout << "hole position: " << endl << trans_center[0][1] << "   " << trans_center[0][2] << endl;
      // cout << "Angle diff: " << endl << center_vec << endl;
      // cout << "minor_length: " << minor_length << endl;
      // vert_dir << 0,1;
      // hori_dir << 1,0;
      // vert_dir = rot * vert_dir;
      // hori_dir = rot * hori_dir;
      // double xstep_size = BEAM_STEPSIZE * vert_dir(0);
      // double ystep_size = BEAM_STEPSIZE * vert_dir(1);
      // // // cout << "Center Dir: " << trans_center[1][0] << "  "
      // // //                      << trans_center[1][1] << "  "
      // // //                      << trans_center[1][2] << endl;
      // // cout << "Projected Ray: " << endl << projected_ray << endl;

      // bool collided = false;
      
      // double estX = 0;
      // double estY = 0;
      // Eigen::Vector2d meanEst;
      // Eigen::Matrix2d invRot = rot.inverse();
      // double h = minor_length;
      // double w = circle_radius;
      // double r = BEAM_RADIUS;
      // double x1 = projected_ray(0);
      // double y1 = projected_ray(1);
      // while (abs(x1) <= w && abs(y1) <= h) {
      //   if (cEllipse.circleEllipseIntersection(BEAM_RADIUS, x1, y1, w, h) || 
      //       cEllipse.circleEllipseIntersection(BEAM_RADIUS, x1, y1 + depth_offset, w, h)) {
      //     meanEst << x1, y1;
      //     // cout << "y top   ";
      //     break;
      //   }
      //   x1 += xstep_size;
      //   y1 += ystep_size;
      // }
      // x1 = projected_ray(0) - xstep_size;
      // y1 = projected_ray(1) - ystep_size;
      // while (abs(x1) <= w && abs(y1) <= h) {
      //   if (cEllipse.circleEllipseIntersection(BEAM_RADIUS, x1, y1, w, h) || 
      //         cEllipse.circleEllipseIntersection(BEAM_RADIUS, x1, y1 + depth_offset, w, h)) {
      //       meanEst(0) += x1;
      //       meanEst(1) += y1;
      //       meanEst /= 2;
      //       meanEst = invRot * meanEst;
      //       estY = meanEst(1);
      //       // cout << "y bot   ";
      //       break;
      //   } 
      //   x1 -= xstep_size;
      //   y1 -= ystep_size;
      // }
      // xstep_size = BEAM_STEPSIZE * hori_dir(0);
      // ystep_size = BEAM_STEPSIZE * hori_dir(1);
      // x1 = projected_ray(0);
      // y1 = projected_ray(1);
      // while (abs(x1) <= w && abs(y1) <= h) {
      //   if (cEllipse.circleEllipseIntersection(BEAM_RADIUS, x1, y1, w, h) || 
      //       cEllipse.circleEllipseIntersection(BEAM_RADIUS, x1, y1 + depth_offset, w, h)) {
      //     meanEst << x1, y1;
      //     // cout << "x right   ";
      //     break;
      //   }
      //   x1 += xstep_size;
      //   y1 += ystep_size;
      // }
      // x1 = projected_ray(0) - xstep_size;
      // y1 = projected_ray(1) - ystep_size;
      // while (abs(x1) <= w && abs(y1) <= h) {
      //   if (cEllipse.circleEllipseIntersection(BEAM_RADIUS, x1, y1, w, h) || 
      //       cEllipse.circleEllipseIntersection(BEAM_RADIUS, x1, y1 + depth_offset, w, h)) {
      //     meanEst(0) += x1;
      //     meanEst(1) += y1;
      //     meanEst /= 2;
      //     meanEst = invRot * meanEst;
      //     estX = meanEst(0);
      //     // cout << "x left" << endl;
      //     break;
      //   }
      //   x1 -= xstep_size;
      //   y1 -= ystep_size;
      // }
      // //cout << "X: " << estX << "    Y: " << estY << endl;
      // //cout << "Depth offset: " << depth_offset << endl;
      // cumError += sqrt(SQ(estX) + SQ(estY));

    }
  }

  // cumError /= rate;
  rate /= numParticles;
  // rate2 /= numParticles;
  // cout << "Average Error: " << cumError << endl;
  cout << "Succ Rate: " << rate << endl;
  // cout << "Succ Rate2: " << rate2 << endl;
  return rate;
}









int main()
{
 //  vector<vec4x3> mesh = importSTL("boeing_part_binary.stl");
 //  int numParticles = 500; // number of particles
 //  double Xstd_ob = 0.001;
 //  double Xstd_tran = 0.0035;
 //  double Xstd_scatter = 0.0001;
 //  //double voxel_size = 0.0005; // voxel size for distance transform.
 //  int num_voxels[3] = { 200,200,200 };
 //  //double range = 0.1; //size of the distance transform
 //  double R = 0.001; // radius of the touch probe

 //  double cube_para[3] = { 6, 4, 2 }; // cube size: 6m x 4m x 2m with center at the origin.
 //  //double range[3][2] = { {-3.5, 3.5}, {-2.5, 2.5}, {-1.5, 1.5} };
 //  cspace X_true = { 2.12, 1.388, 0.818, Pi / 6 + Pi / 400, Pi / 12 + Pi / 220, Pi / 18 - Pi / 180 }; // true state of configuration
 //  //cspace X_true = { 0, 0, 0.818, 0, 0, 0 }; // true state of configuration
 //  cout << "True state: " << X_true[0] << ' ' << X_true[1] << ' ' << X_true[2] << ' ' 
	//    << X_true[3] << ' ' << X_true[4] << ' ' << X_true[5] << endl;
 //  cspace b_Xprior[2] = { { 2.11, 1.4, 0.81, Pi / 6, Pi / 12, Pi / 18 },
	// 									 { 0.03, 0.03, 0.03, Pi / 180, Pi / 180, Pi / 180 } }; // our prior belief
 //  //cspace b_Xprior[2] = { { 0, 0, 0.81, 0, 0, 0 },
 //  //									 { 0.001, 0.001, 0.001, Pi / 3600, Pi / 3600, Pi / 3600 } }; // our prior belief

 //  particleFilter pfilter(numParticles, b_Xprior, Xstd_ob, Xstd_tran, Xstd_scatter, R);
 //  distanceTransform *dist_transform = new distanceTransform(num_voxels);
 //  //dist_transform->build(cube_para);

 //  int N_Measure = 60; // total number of measurements
 //  double M_std = 0.000; // measurement error
 //  double M[2][3]; // measurement
 //  cspace particles_est;
 //  cspace particles_est_stat;
 //  double particle_est_diff;

 //  std::random_device generator;
 //  std::uniform_real_distribution<double> distribution(0, 1);

 //  double pstart[3];
 //  normal_distribution<double> dist(0, M_std);
 //  for (int i = 0; i < N_Measure; i++) {
	// // generate measurement in fixed frame, then transform it to particle frame.
	// //1.117218  0.043219  0.204427
	// if (i % 3 == 0)
	//   {
	// 	M[1][0] = -1; M[1][1] = 0; M[1][2] = 0;
	// 	pstart[0] = 2;
	// 	pstart[1] = distribution(generator) * 0.07 + 0.15 + dist(generator);
	// 	pstart[2] = distribution(generator) * 0.1 + 0.05 + dist(generator);
	// 	if (getIntersection(mesh, pstart, M[1], M[0]) == 0)
	// 	  cout << "err" << endl;
	// 	M[0][0] += R + dist(generator);	
	//   }
	// else if (i % 3 == 1)
	//   {
	// 	M[1][0] = 0; M[1][1] = -1; M[1][2] = 0;
	// 	pstart[0] = distribution(generator) * 0.15 + 1.38 + dist(generator);
	// 	pstart[1] = 1;
	// 	pstart[2] = distribution(generator) * 0.1 + 0.05 + dist(generator);
	// 	if (getIntersection(mesh, pstart, M[1], M[0]) == 0)
	// 	  cout << "err" << endl;
	// 	M[0][1] += R + dist(generator);
	//   }
	// else
	//   {
	// 	M[1][0] = 0; M[1][1] = 0; M[1][2] = -1;
	// 	pstart[0] = distribution(generator) * 1 + 0.2 + dist(generator);
	// 	pstart[1] = distribution(generator) * 0.03 + 0.02 + dist(generator);
	// 	pstart[2] = 1;
	// 	if (getIntersection(mesh, pstart, M[1], M[0]) == 0)
	// 	  cout << "err" << endl;
	// 	M[0][2] += R + dist(generator);
			
	//   }
	// Transform(M, X_true, M);
	// //rotationMatrix(X_true, rotationM);
	// //multiplyM(rotationM, M[0], tempM);
	// //double transition[3] = { X_true[0], X_true[1], X_true[2] };
	// //addM(tempM, transition, M[0]);
	// ///*multiplyM(rotationM, M[1], tempM);
	// //M[1][0] = tempM[0];
	// //M[1][1] = tempM[1];
	// //M[1][2] = tempM[2];*/

	// cout << "Observation " << i << " : touch at " << M[0][0] << " " << M[0][1] << " " << M[0][2] << endl;
	// cout << "Theoretic distance: " << testResult(mesh, X_true, M, R) << endl;
	// auto tstart = chrono::high_resolution_clock::now();
	// pfilter.addObservation(M, mesh, dist_transform, i); // update particles
	// //pfilter.addObservation(M, cube_para, i);
	// pfilter.estimateGaussian(particles_est, particles_est_stat);
	// auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(chrono::high_resolution_clock::now() - tstart);
	// particle_est_diff = 0;
	// for (int k = 0; k < particleFilter::cdim; k++) {
	//   particle_est_diff += SQ(particles_est[k] - X_true[k]);
	// }
	// particle_est_diff /= particleFilter::cdim;
	// particle_est_diff = sqrt(particle_est_diff);
	// cout << "est: ";
	// for (int k = 0; k < particleFilter::cdim; k++) {
	//   cout << particles_est[k] << ' ';
	// }
	// cout << endl;
	// cout << "Real distance: " << testResult(mesh, particles_est, M, R) << endl;
	// cout << "Diff: " << particle_est_diff << endl;
	// cout << "Var: ";
	// for (int k = 0; k < particleFilter::cdim; k++) {
	//   cout << particles_est_stat[k] << ' ';
	// }
	// cout << endl;
	// cout << "Time: " << diff.count() << " milliseconds." << endl << endl;
 //  }
 //  delete (dist_transform);
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

/*
 * Check if the center of a voxel is within the object
 * defined by mesh arrays.
 * Input: mesh arrays
 *        voxel center
 * Output: 1 if inside
 *         0 if outside
 */
int checkInObject(vector<vec4x3> &mesh, double voxel_center[3])
{
  int countIntersections = 0;
  int num_mesh = int(mesh.size());
  double dir[3] = { 1,0,0 };
  double vert0[3], vert1[3], vert2[3];
  double *t = new double; 
  double *u = new double; 
  double *v = new double;
  std::unordered_set<double> hashset;
  for (int i = 0; i < num_mesh; i++)
	{
	  vert0[0] = mesh[i][1][0];
	  vert0[1] = mesh[i][1][1];
	  vert0[2] = mesh[i][1][2];
	  vert1[0] = mesh[i][2][0];
	  vert1[1] = mesh[i][2][1];
	  vert1[2] = mesh[i][2][2];
	  vert2[0] = mesh[i][3][0];
	  vert2[1] = mesh[i][3][1];
	  vert2[2] = mesh[i][3][2];
	  if (intersect_triangle(voxel_center, dir, vert0, vert1, vert2, t, u, v) == 1)
		{
		  if (hashset.find(*t) == hashset.end())
			{
			  hashset.insert(*t);
			  countIntersections++;
			}
		}
			
	}
  if (countIntersections % 2 == 0)
	{
	  return 0;
	}
  delete t, u, v;
  return 1;
}

/*
 * Find the intersection point between a ray segment and meshes
 * Input: mesh: mesh arrays
 *        pstart: start point
 *        pend: end point
 * Output: 1 if intersect
 *         0 if not
 */
int getIntersectionSeg(vector<vec4x3> &mesh, double pstart[3], double pend[3])
{
  int num_mesh = int(mesh.size());
  double vert0[3], vert1[3], vert2[3];
  double *t = new double;
  double *u = new double;
  double *v = new double;
  double dir[3];
  dir[0] = pend[0] - pstart[0];
  dir[1] = pend[1] - pstart[1];
  dir[2] = pend[2] - pstart[2];
  double seg_length = sqrt(SQ(dir[0]) + SQ(dir[1]) + SQ(dir[2]));
  dir[0] /= seg_length;
  dir[1] /= seg_length;
  dir[2] /= seg_length;
  for (int i = 0; i < num_mesh; i++)
  {
    vert0[0] = mesh[i][1][0];
    vert0[1] = mesh[i][1][1];
    vert0[2] = mesh[i][1][2];
    vert1[0] = mesh[i][2][0];
    vert1[1] = mesh[i][2][1];
    vert1[2] = mesh[i][2][2];
    vert2[0] = mesh[i][3][0];
    vert2[1] = mesh[i][3][1];
    vert2[2] = mesh[i][3][2];
    if (intersect_triangle(pstart, dir, vert0, vert1, vert2, t, u, v) == 1 && *t < seg_length)
    {
      delete t, u, v;
      // cout << "   " << dir[0] << "  " << dir[1] << "  " << dir[2] << endl;
      // cout << "   " << pstart[0] << "  " << pstart[1] << "  " << pstart[2] << endl;
      return 1;
    }

  }
  delete t, u, v;
  
  return 0;
}

/*
 * Find the intersection point between a ray and meshes
 * Input: mesh: mesh arrays
 * 	      pstart: start point
 * 	      dir: ray direction
 * 	      intersection: intersection point
 * Output: 1 if intersect
 *         0 if not
 */
int getIntersection(vector<vec4x3> &mesh, double pstart[3], double dir[3], double intersection[3])
{
  int num_mesh = int(mesh.size());
  double vert0[3], vert1[3], vert2[3];
  double *t = new double;
  double *u = new double;
  double *v = new double;
  double tMin = 100000;
  for (int i = 0; i < num_mesh; i++)
	{
	  vert0[0] = mesh[i][1][0];
	  vert0[1] = mesh[i][1][1];
	  vert0[2] = mesh[i][1][2];
	  vert1[0] = mesh[i][2][0];
	  vert1[1] = mesh[i][2][1];
	  vert1[2] = mesh[i][2][2];
	  vert2[0] = mesh[i][3][0];
	  vert2[1] = mesh[i][3][1];
	  vert2[2] = mesh[i][3][2];
	  if (intersect_triangle(pstart, dir, vert0, vert1, vert2, t, u, v) == 1 && *t < tMin)
		{
		  tMin = *t;
		}

	}
  delete t, u, v;
  if (tMin == 100000)
		return 0;
  for (int i = 0; i < 3; i++)
	{
	  intersection[i] = pstart[i] + dir[i] * tMin;
	}
	
  return 1;
}

/*
 * Calculate the distance between touch probe and object using 
 * mean estimated configuration after each update
 * Input: mesh: mesh arrays
 *        config: estimated mean configuration
 *        touch: touch point in particle frame
 *        R: radius of the touch probe
 * Output: distance
 */
double testResult(vector<vec4x3> &mesh, cspace config, double touch[2][3], double R)
{
  double inv_touch[2][3];
  inverseTransform(touch, config, inv_touch);
  int num_mesh = int(mesh.size());
  double vert0[3], vert1[3], vert2[3];
  double *t = new double;
  double *u = new double;
  double *v = new double;
  double tMin = 100000;
  for (int i = 0; i < num_mesh; i++)
	{
	  vert0[0] = mesh[i][1][0];
	  vert0[1] = mesh[i][1][1];
	  vert0[2] = mesh[i][1][2];
	  vert1[0] = mesh[i][2][0];
	  vert1[1] = mesh[i][2][1];
	  vert1[2] = mesh[i][2][2];
	  vert2[0] = mesh[i][3][0];
	  vert2[1] = mesh[i][3][1];
	  vert2[2] = mesh[i][3][2];
	  if (intersect_triangle(inv_touch[0], inv_touch[1], vert0, vert1, vert2, t, u, v) == 1 && *t < tMin)
		{
		  tMin = *t;
		}
	}
  delete t, u, v;
  if (tMin == 100000)
		return 0;

  return tMin - R;
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
  for (int i = 0; i < particleFilter::cdim; i++) {
		s += floor(config[i] / DISPLACE_INTERVAL);
		s += ":";
  }
  if (set->find(s) == set->end()) {
		set->insert(s);
		return 1;
  }
  return 0;
}

/*
 * Raytrace checker. Check obstacle along the ray
 * Input: mesh: mesh arrays
 *        config: estimated mean configuration
 *        start: safepoint of the joint
 *        dist: distance between center of touch probe and object
 * Output: 1 if obstacle exists
 */
int checkObstacles(vector<vec4x3> &mesh, cspace config, double start[2][3], double dist)
{
  return checkObstacles(mesh, config, start, ARM_LENGTH, dist);
}
int checkObstacles(vector<vec4x3> &mesh, cspace config, double start[2][3], double check_length, double dist)
{
  double inv_start[2][3];
  int countIntersections = 0;
  inverseTransform(start, config, inv_start);
  int num_mesh = int(mesh.size());
  double vert0[3], vert1[3], vert2[3]; 
  double *t = new double;
  double *u = new double;
  double *v = new double;
  double tMin = 100000;
  Eigen::Vector3d normal_dir;
  Eigen::Vector3d ray_length;
  double inside_length;
  std::unordered_set<double> hashset;
  for (int i = 0; i < num_mesh; i++)
	{
	  vert0[0] = mesh[i][1][0];
	  vert0[1] = mesh[i][1][1];
	  vert0[2] = mesh[i][1][2];
	  vert1[0] = mesh[i][2][0];
	  vert1[1] = mesh[i][2][1];
	  vert1[2] = mesh[i][2][2];
	  vert2[0] = mesh[i][3][0];
	  vert2[1] = mesh[i][3][1];
	  vert2[2] = mesh[i][3][2];
	  if (intersect_triangle(inv_start[0], inv_start[1], vert0, vert1, vert2, t, u, v) == 1)
		{
		  if (*t < tMin)
			{
			  tMin = *t;
			  normal_dir << mesh[i][0][0], mesh[i][0][1], mesh[i][0][2];
			  inside_length = check_length - tMin;
			  ray_length << inside_length * inv_start[1][0], inside_length * inv_start[1][1], inside_length * inv_start[1][2];
			}
		  if (hashset.find(*t) == hashset.end())
			{
			  hashset.insert(*t);
			  countIntersections++;
			}
		}
	}
  delete t, u, v;
  if (countIntersections % 2 == 1)
	{
	  return 1;
	}
  if (tMin >= check_length)
		return 0;
  else if (dist < 0)
	{
	  double inter_dist = normal_dir.dot(ray_length);
	  //cout << "inter_dist: " << inter_dist << endl;
	  if (inter_dist >= dist - EPSILON && inter_dist <= dist + EPSILON)
			return 0;
	}
		
  return 1;
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

int particleFilter::getNumParticles() {
	return fullStateFilter.numParticles;
}

