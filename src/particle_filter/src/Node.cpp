#include <string.h>
#include <iostream>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <chrono>
#include "definitions.h"
#include "tribox.h"
#include "raytri.h"
#include "circleEllipse.h"
#include "distanceTransformNew.h"
#include "matrix.h"
#include "Node.h"


using namespace std;

const int Node::cdim;
int total_time = 0;
int converge_count = 0;
double TRUE_STATE[6] = {0.3, 0.3, 0.3, 0.5, 0.7, 0.5};
// Construction for root
Node::Node(int n_particles, cspace b_init[2]):numParticles(n_particles), type(0) {
  maxNumParticles = numParticles;
  numUpdates = false;
  fullParticles.resize(numParticles);
  fullParticlesPrev.resize(numParticles);
  particles.resize(numParticles);
  particlesPrev.resize(numParticles);
  createParticles(b_init, numParticles, 1);
  fulldim = cdim;
  Parent *parentPtr = new Parent(this, 0, 0);
  std::vector<Parent *> p;
  p.push_back(parentPtr);
  child.push_back(new Node(numParticles, p, 1));
  cspace prior[2] = {{0,0,0,1.22,0,0},{0,0,0,0.0005,0.0005,0.0005}};
  child.push_back(new Node(numParticles, p, prior));
  cspace prior2[2] = {{0,-0.025,0,0,-0.025,0.23},{0,0,0,0.0005,0.0005,0.0005}};
  child.push_back(new Node(numParticles, p, prior2));
  std::vector<Parent *> topP;
  topP.push_back(new Parent(child[0], 0, 0));
  topP.push_back(new Parent(child[1], 0.23, 0.001));
  Node *topEdge = new Node(numParticles, topP, 2);
  child[0]->child.push_back(topEdge);
  child[1]->child.push_back(topEdge);
  
  Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)particlesPrev.data(), fulldim, numParticles);
  Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
  full_cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
  mat = Eigen::Map<Eigen::MatrixXd>((double *)particlesPrev.data(), cdim, numParticles);
  mat_centered = mat.colwise() - mat.rowwise().mean();
  cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
  // cout << full_cov_mat << endl;

}
// Construction for initial datums
Node::Node (int n_particles, std::vector<Parent *> &p, cspace b_init[2]):numParticles(n_particles), type(2)  {
  maxNumParticles = numParticles;
  numUpdates = false;
  fullParticles.resize(numParticles);
  fullParticlesPrev.resize(numParticles);
  particles.resize(numParticles);
  particlesPrev.resize(numParticles);
  int size = p.size();
  fulldim = cdim * (size + 1);
  for (int i = 0; i < size; i ++) {
    parent.push_back(p[i]);
  }
  type = 2;

  createParticles(b_init, numParticles, 0);

  double* tmpParticles = new double[numParticles * fulldim];
  for(int i = 0; i < numParticles; ++i) {
    for (int j = 0; j < fulldim; j ++) {
      tmpParticles[i * fulldim + j] = fullParticles[i][j];
    }
  }

  Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(tmpParticles, fulldim, numParticles);
  Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
  full_cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
  mat = Eigen::Map<Eigen::MatrixXd>((double *)particlesPrev.data(), cdim, numParticles);
  mat_centered = mat.colwise() - mat.rowwise().mean();
  cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));

  delete [] tmpParticles;
}
// Construction for other datums
Node::Node(int n_particles, std::vector<Parent *> &p, int t):numParticles(n_particles), type(t) {
  maxNumParticles = numParticles;
  numUpdates = false;
  fullParticles.resize(numParticles);
  fullParticlesPrev.resize(numParticles);
  particles.resize(numParticles);
  particlesPrev.resize(numParticles);
  int size = p.size();
  fulldim = cdim * (size + 1);
  for (int i = 0; i < size; i ++) {
    parent.push_back(p[i]);
    // p[i]->node->child.push_back(this);
  }

  createParticles();

  double* tmpParticles = new double[numParticles * fulldim];
  for(int i = 0; i < numParticles; ++i) {
    for (int j = 0; j < fulldim; j ++) {
      tmpParticles[i * fulldim + j] = fullParticles[i][j];
    }
  }

  Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(tmpParticles, fulldim, numParticles);
  Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
  full_cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
  mat = Eigen::Map<Eigen::MatrixXd>((double *)particles.data(), cdim, numParticles);
  mat_centered = mat.colwise() - mat.rowwise().mean();
  cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));

  delete [] tmpParticles;
}


void Node::addDatum(double d, double td) {
  Parent *p = new Parent(this, d, td);

}

/*
 * Create initial particles at start
 * Input: particles
 *        b_Xprior: prior belief
 *        n_partcles: number of particles
 * output: none
 */
void Node::createParticles(cspace b_Xprior[2], int n_particles, int isRoot)
{
  std::random_device rd;
  std::normal_distribution<double> dist(0, 1);
  if (isRoot == 1) {
    for (int i = 0; i < n_particles; i++) {
      fullParticlesPrev[i].resize(cdim);
      for (int j = 0; j < cdim; j++) {
        particlesPrev[i][j] = b_Xprior[0][j] + b_Xprior[1][j] * (dist(rd));
        fullParticlesPrev[i][j] = particlesPrev[i][j];
      }
    }
    cout << "Finish creating root!" << endl;
  } else {
    std::uniform_int_distribution<> distRoot(0, parent[0]->node->numParticles - 1);
    cspace relativeConfig, baseConfig;

    for (int i = 0; i < n_particles; i++) {
      fullParticlesPrev[i].resize((parent.size() + 1) * cdim);
      for (int j = 0; j < cdim; j++) {
        relativeConfig[j] = b_Xprior[0][j] + b_Xprior[1][j] * (dist(rd));
      }
      baseConfig = parent[0]->node->particles[int(distRoot(rd))];
      // priorParticles[0][i] = baseConfig;
      copyParticles(baseConfig, fullParticlesPrev[i], cdim);
      if (type == 1)
        transFrameConfig(baseConfig, relativeConfig, particlesPrev[i]);
      else
        transPointConfig(baseConfig, relativeConfig, particlesPrev[i]);
      copyParticles(particlesPrev[i], fullParticlesPrev[i], 0);
    }
    cout << "Finish creating edges!" << endl;
  }
  particles = particlesPrev;
  fullParticles = fullParticlesPrev;
}

/*
 * Create initial particles at start
 * Input: particles
 *        b_Xprior: prior belief
 *        n_partcles: number of particles
 * output: none
 */
void Node::createParticles()
{
  int size = parent.size();
  if (size == 0) return;
  if (type == 1) {
    std::random_device rd;
    std::uniform_int_distribution<> distRoot(0, parent[0]->node->numParticles - 1);
    cspace relativeConfig, baseConfig;
    relativeConfig[0] = 1.22;
    relativeConfig[1] = -0.025;
    relativeConfig[2] = 0;
    relativeConfig[3] = 0;
    relativeConfig[4] = 0;
    relativeConfig[5] = Pi;
    for (int i = 0; i < numParticles; i++) {
      int indexRoot = int(distRoot(rd));
      fullParticlesPrev[i].resize((parent.size() + 1) * cdim);
      baseConfig = parent[0]->node->particles[indexRoot];
      copyParticles(baseConfig, fullParticlesPrev[i], cdim);
      transFrameConfig(baseConfig, relativeConfig, particlesPrev[i]);
      //TEMP:
      if (particlesPrev[i][5] < 0)  particlesPrev[i][5] += 2 * Pi;
      copyParticles(particlesPrev[i], fullParticlesPrev[i], 0);
      // for (int j = 0; j < 6; j ++) {
      //   cout << particlesPrev[i][j] << ", ";
      // }
      // cout << endl;
    }
    cout << "Finish creating plane!" << endl;
  } else if (type == 2) {
    Parent *plane = parent[0];
    Parent *edge = parent[1];
    
    double edgeTol = edge->tol;
    double edgeOffSet = edge->offset;
    std::random_device rd;
    std::uniform_int_distribution<> distEdge(0, edge->node->numParticles - 1);
    std::uniform_int_distribution<> distPlane(0, plane->node->numParticles - 1);
    std::uniform_real_distribution<double> dist(-1, 1);

    for (int i = 0; i < numParticles; i ++) {
      int indexEdge = int(distEdge(rd));
      fullParticlesPrev[i].resize((parent.size() + 1) * cdim);
      cspace configEdge = edge->node->particles[indexEdge];
      copyParticles(configEdge, fullParticlesPrev[i], 2 * cdim);
      Eigen::Vector3d pa, pb; 
      pa << configEdge[0], configEdge[1], configEdge[2];
      pb << configEdge[3], configEdge[4], configEdge[5];
      int indexPlane = int(distPlane(rd));
      cspace configPlane = plane->node->particles[indexPlane];
      copyParticles(configPlane, fullParticlesPrev[i], cdim);
      Eigen::Vector3d pa_prime, pb_prime;
      inverseTransform(pa, configPlane, pa_prime);
      inverseTransform(pb, configPlane, pb_prime);
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
      Transform(pa_prime, configPlane, pa);
      Transform(pb_prime, configPlane, pb);
      particlesPrev[i][0] = pa(0);
      particlesPrev[i][1] = pa(1);
      particlesPrev[i][2] = pa(2);
      particlesPrev[i][3] = pb(0);
      particlesPrev[i][4] = pb(1);
      particlesPrev[i][5] = pb(2);
      copyParticles(particlesPrev[i], fullParticlesPrev[i], 0);
    }
  } else {
    Parent *plane = parent[0];
    Parent *edge1 = parent[1];
    Parent *edge2 = parent[2];

    double edgeTol1 = edge1->tol;
    double edgeOffSet1 = edge1->offset;
    double edgeTol2 = edge2->tol;
    double edgeOffSet2 = edge2->offset;

    std::random_device rd;
    std::uniform_int_distribution<> distEdge1(0, edge1->node->numParticles - 1);
    std::uniform_int_distribution<> distEdge2(0, edge2->node->numParticles - 1);
    std::uniform_int_distribution<> distPlane(0, plane->node->numParticles - 1);
    std::uniform_real_distribution<double> dist(-1, 1);
    for (int i = 0; i < numParticles; i ++) {
      int indexEdge1 = int(distEdge1(rd));
      int indexEdge2 = int(distEdge2(rd));
      cspace configEdge1 = edge1->node->particles[indexEdge1];
      cspace configEdge2 = edge2->node->particles[indexEdge2];
      Eigen::Vector3d pa1, pb1, pa2, pb2;
      pa1 << configEdge1[0], configEdge1[1], configEdge1[2];
      pb1 << configEdge1[3], configEdge1[4], configEdge1[5];
      pa2 << configEdge2[0], configEdge2[1], configEdge2[2];
      pb2 << configEdge2[3], configEdge2[4], configEdge2[5];
      int indexPlane = int(distPlane(rd));
      cspace configPlane = plane->node->particles[indexPlane];
      Eigen::Vector3d pa1_prime, pb1_prime, pa2_prime, pb2_prime;
      inverseTransform(pa1, configPlane, pa1_prime);
      inverseTransform(pb1, configPlane, pb1_prime);
      inverseTransform(pa2, configPlane, pa2_prime);
      inverseTransform(pb2, configPlane, pb2_prime);
      double td = dist(rd) * edgeTol1;
      pa1_prime(1) = 0;
      pb1_prime(1) = 0;
      pa2_prime(1) = 0;
      pb2_prime(1) = 0;
      Eigen::Vector3d normVec;
      normVec << (pb1_prime(2) - pa1_prime(2)), 0, (pa1_prime(0) - pb1_prime(0));
      normVec.normalize();
      normVec *= (edgeOffSet1 + td);
      pa1_prime(0) += normVec(0);
      pb1_prime(0) += normVec(0);
      pa1_prime(2) += normVec(2);
      pb1_prime(2) += normVec(2);
      td = dist(rd) * edgeTol2;
      normVec << (pb2_prime(2) - pa2_prime(2)), 0, (pa2_prime(0) - pb2_prime(0));
      normVec.normalize();
      normVec *= (edgeOffSet2 + td);
      pa2_prime(0) += normVec(0);
      pb2_prime(0) += normVec(0);
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
      pi_prime(1) = 0;
      pi_prime(2) = dividend.determinant() / divisor.determinant();
      dir_prime << 0, 1, 0; origin_prime << 0, 0, 0;
      Transform(pi_prime, configPlane, pi);
      Transform(dir_prime, configPlane, dir);
      Transform(origin_prime, configPlane, origin);
      dir -= origin;
      particlesPrev[i][0] = pi(0);
      particlesPrev[i][1] = pi(1);
      particlesPrev[i][2] = pi(2);
      particlesPrev[i][3] = dir(0);
      particlesPrev[i][4] = dir(1);
      particlesPrev[i][5] = dir(2);
    }
  }
  particles = particlesPrev;
  fullParticles = fullParticlesPrev;
}
// sample a config from the particles uniformly.
void Node::sampleConfig(cspace &config) {
  std::random_device rd;
  std::uniform_int_distribution<> dist(0, numParticles - 1);
  config = particlesPrev[int(dist(rd))];
}

void Node::estimateGaussian(cspace &x_mean, cspace &x_est_stat) {
  cout << "Estimated Mean: ";
  for (int k = 0; k < cdim; k++) {
  x_mean[k] = 0;
  for (int j = 0; j < numParticles; j++) {
    x_mean[k] += particlesPrev[j][k];
  }
  x_mean[k] /= numParticles;
  cout << x_mean[k] << "  ";
  }
  cout << endl;
  cout << "Estimated Std: ";
  for (int k = 0; k < cdim; k++) {
  x_est_stat[k] = 0;
  for (int j = 0; j < numParticles; j++) {
    x_est_stat[k] += SQ(particlesPrev[j][k] - x_mean[k]);
  }
  x_est_stat[k] = sqrt(x_est_stat[k] / numParticles);
  cout << x_est_stat[k] << "  ";
  }
  cout << endl;

}

void Node::getAllParticles(Particles &particles_dest)
{
  particles_dest = particlesPrev;
}

void Node::getPriorParticles(Particles &particles_dest, int idx)
{
  particles_dest.resize(numParticles);
  for (int i = 0; i < numParticles; i ++) {
    for (int j = 0; j < cdim; j ++) {
      particles_dest[i][j] = fullParticlesPrev[i][idx + j];
    }
  }
}

void Node::resampleParticles(Particles &rootParticles, Particles &rootParticlesPrev, int n, double *W)
{
  double *Cum_sum = new double[n];
  Cum_sum[0] = W[0];
  std::random_device generator;
  std::uniform_real_distribution<double> rd(0, 1);
  for (int i = 1; i < n; i++)
  {
    Cum_sum[i] = Cum_sum[i - 1] + W[i];
    // cout << Cum_sum[i] << endl;
  }
  double t;
  for (int i = 0; i < n; i++)
  {
     t = rd(generator);
     for (int j = 0; j < n; j++)
     {
       if (j == 0 && t <= Cum_sum[0])
       {
         rootParticles[i][0] = rootParticlesPrev[0][0];
         rootParticles[i][1] = rootParticlesPrev[0][1];
         rootParticles[i][2] = rootParticlesPrev[0][2];
         rootParticles[i][3] = rootParticlesPrev[0][3];
         rootParticles[i][4] = rootParticlesPrev[0][4];
         rootParticles[i][5] = rootParticlesPrev[0][5];
         break;
       }
       else if (Cum_sum[j - 1] < t && t <= Cum_sum[j])
       {
         rootParticles[i][0] = rootParticlesPrev[j][0];
         rootParticles[i][1] = rootParticlesPrev[j][1];
         rootParticles[i][2] = rootParticlesPrev[j][2];
         rootParticles[i][3] = rootParticlesPrev[j][3];
         rootParticles[i][4] = rootParticlesPrev[j][4];
         rootParticles[i][5] = rootParticlesPrev[j][5];
         break;
       }
     }
  }
  rootParticlesPrev = rootParticles;
}

void Node::sampleParticles() {
  std::random_device rd;
  std::normal_distribution<double> dist(0, 1);
  std::uniform_real_distribution<double> distribution(0, numParticles);
  Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)fullParticlesPrev.data(), fulldim, numParticles);
  Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
  full_cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
  double coeff = pow(numParticles, -0.2) * 0.87055/1.2155/1.2155;
  Eigen::MatrixXd H_cov = coeff * full_cov_mat;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(H_cov);
  Eigen::MatrixXd rot = eigenSolver.eigenvectors(); 
  Eigen::VectorXd scl = eigenSolver.eigenvalues();

  for (int j = 0; j < cdim; j++) {
    scl(j, 0) = sqrt(scl(j, 0));
  }
  Eigen::VectorXd samples(cdim, 1);
  Eigen::VectorXd rot_sample(cdim, 1);
  for (int i = 0; i < numParticles; i ++) {
    int idx = int(floor(distribution(rd)));

    for (int j = 0; j < cdim; j++) {
      samples(j, 0) = scl(j, 0) * dist(rd);
    }
    rot_sample = rot*samples;
    for (int j = 0; j < cdim; j++) {
      /* TODO: use quaternions instead of euler angles */
      particlesPrev[i][j] = particles[i][j] + rot_sample(j, 0);
    }
  }
  particles = particlesPrev;
}

void Node::propagate() {
  numUpdates = !numUpdates;
  if (type == 0) { // Root
    int n = child.size();
    for (int i = 0; i < n; i ++) {
      Node *ch = child[i];
      if (ch->numUpdates == numUpdates) continue;
      if (i == 0) {
        int chNumParticles = ch->numParticles;
        cspace tempConfig, tConfig;
        cspace relativeConfig, baseConfig;
        cspace curMean, curVar;
        Particles transParticles;
        transParticles.resize(numParticles);
        relativeConfig[0] = 1.22;
        relativeConfig[1] = -0.025;
        relativeConfig[2] = 0;
        relativeConfig[3] = 0;
        relativeConfig[4] = 0;
        relativeConfig[5] = Pi;

        // // estimateGaussian(curMean, curVar);
        for (int i = 0; i < numParticles; i ++ ) {
          transFrameConfig(particles[i], relativeConfig, transParticles[i]);
        }
        Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)transParticles.data(), cdim, numParticles);
        Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
        Eigen::MatrixXd cov = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
        double coeff = pow(numParticles, -0.2) * 0.87055;
        cov = coeff * cov;
        double A = 1.0 / (sqrt(pow(2 * Pi, 6) * cov.determinant()));
        Eigen::Matrix<double, 6, 6> B = -0.5 * (cov.inverse());
        double sum = 0;
        double *weight = new double[chNumParticles];
        
        for (int i = 1; i < chNumParticles; i ++ ) {
          tempConfig = ch->particles[i];

          weight[i] = 0;
          Eigen::Matrix<double, 6, 1> x;
          for (int j = 0; j < numParticles; j ++) {
            x << tempConfig[0] - transParticles[j][0], tempConfig[1] - transParticles[j][1], tempConfig[2] - transParticles[j][2],
                 tempConfig[3] - transParticles[j][3], tempConfig[4] - transParticles[j][4], tempConfig[5] - transParticles[j][5];
            weight[i] += A*exp(x.transpose() * B * x);
          }
          sum += weight[i];
        }
        for (int i = 0; i < chNumParticles; i ++ ) {
          weight[i] /= sum;
        }
        resampleParticles(ch->particles, ch->particlesPrev, chNumParticles, weight);
        ch->sampleParticles(); 
        ch->propagate();
      }
    }


  } else if (type == 1) { // Plane
    Node *root = parent[0]->node;
    if (root->numUpdates == numUpdates) return;
    // Particles *rootParticlesPrev = root->particlesPrev;
    // Particles *rootParticles = root->particles;
    int n = root->numParticles;
    cspace tempConfig, tConfig;
    cspace relativeConfig, baseConfig;
    cspace curMean, curVar;
    Particles invParticles;
    invParticles.resize(numParticles);
    relativeConfig[0] = 1.22;
    relativeConfig[1] = -0.025;
    relativeConfig[2] = 0;
    relativeConfig[3] = 0;
    relativeConfig[4] = 0;
    relativeConfig[5] = Pi;
    estimateGaussian(curMean, curVar);
    for (int i = 0; i < numParticles; i ++ ) {
      invTransFrameConfig(curMean, particles[i], invParticles[i]);
    }
    Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)invParticles.data(), cdim, numParticles);
    Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
    Eigen::MatrixXd cov = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
    double coeff = pow(numParticles, -2.0/7.0) * 0.93823;
    cov = coeff * cov;
    Eigen::Matrix3d normalCov;
    normalCov << cov(1, 1), cov(1, 3), cov(1, 5), 
                 cov(3, 1), cov(3, 3), cov(3, 5), 
                 cov(5, 1), cov(5, 3), cov(5, 5);
    cout << normalCov << endl;
    double A = 1.0 / (sqrt(pow(2 * Pi, 3) * normalCov.determinant()));
    Eigen::Matrix3d B = -0.5 * (normalCov.inverse());
    double sum = 0;
    double *weight = new double[n];
    
    for (int i = 1; i < n; i ++ ) {
      tempConfig = root->particles[i];
      transFrameConfig(tempConfig, relativeConfig, tConfig);
      invTransFrameConfig(curMean, tConfig, tConfig);
      weight[i] = 0;
      Eigen::Vector3d x;
      for (int j = 0; j < numParticles; j ++) {
        x << tConfig[1] - invParticles[j][1], tConfig[3] - invParticles[j][3], tConfig[5] - invParticles[j][5];
        weight[i] += A*exp(x.transpose() * B * x);
      }
      sum += weight[i];
      // double diff = sqrt(SQ(tConfig[1]) + SQ(tConfig[3]) + SQ(tConfig[5]));
      // if (diff < 0.1) {
      //   for (int j = 0; j < cdim; j++) {
      //     root->particles[idx][j] = tempConfig[j];
      //   }
      //   idx ++;
      // }
    }
    for (int i = 0; i < n; i ++ ) {
      weight[i] /= sum;
      // cout << weight[i] << endl;
    }
    // int idx = 0;
    // while (idx < n) {
    //   root->sampleConfig(tempConfig);
    //   // cout << "Sampled state: ";
    //   // for (int i = 0; i < 6; i ++) {
    //   //   cout << tempConfig[i] << ", ";
    //   // }
    //   // cout << endl;
    //   transFrameConfig(tempConfig, relativeConfig, tConfig);
    //   invTransFrameConfig(curMean, tConfig, tConfig);
    //   // cout << "Sampled Transformed state: ";
    //   // for (int i = 0; i < 6; i ++) {
    //   //   cout << tConfig[i] << ", ";
    //   // }
    //   // cout << endl;
    //   double diff = sqrt(SQ(tConfig[1]) + SQ(tConfig[3]) + SQ(tConfig[5]));
    //   if (diff < 0.05) {
    //     for (int j = 0; j < cdim; j++) {
    //       root->particles[idx][j] = tempConfig[j];
    //     }
    //     idx ++;
    //   }
    // }
    resampleParticles(root->particles, root->particlesPrev, n, weight);
    root->sampleParticles(); 
    root->propagate();
  } else if (type == 2) { // Edge 
    Node *root = parent[0]->node;
    if (root->type == 0) {
      
      int n = root->numParticles;
      cspace tempConfig, tConfig;
      cspace relativeConfig, baseConfig;
      relativeConfig[0] = 0;
      relativeConfig[1] = 0;
      relativeConfig[2] = 0;
      relativeConfig[3] = 1.22;
      relativeConfig[4] = 0;
      relativeConfig[5] = 0;

      Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)particles.data(), cdim, numParticles);
      Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
      Eigen::MatrixXd cov = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
      double coeff = pow(numParticles, -2.0/8.0) * 0.90360 * 100;
      cov = coeff * cov;
      Eigen::Matrix4d normalCov;
      normalCov << cov(1, 1), cov(1, 2), cov(1, 4), cov(1, 5), 
                   cov(2, 1), cov(2, 2), cov(2, 4), cov(2, 5), 
                   cov(4, 1), cov(4, 2), cov(4, 4), cov(4, 5), 
                   cov(5, 1), cov(5, 2), cov(5, 4), cov(5, 5);
      cout << normalCov << endl;
      double A = 1.0 / (sqrt(pow(2 * Pi, 4) * normalCov.determinant()));
      Eigen::Matrix4d B = -0.5 * (normalCov.inverse());
      double sum = 0;
      double *weight = new double[n];
      
      for (int i = 1; i < n; i ++ ) {
        tempConfig = root->particles[i];
        transPointConfig(tempConfig, relativeConfig, tConfig);
        weight[i] = 0;
        Eigen::Vector4d x;
        // cout << tempConfig[1] << endl;
        for (int j = 0; j < numParticles; j ++) {
          x << tConfig[1] - particles[j][1], tConfig[2] - particles[j][2], tConfig[4] - particles[j][4], tConfig[5] - particles[j][5];
          weight[i] += A*exp(x.transpose() * B * x);
        }
        sum += weight[i];
        // double diff = sqrt(SQ(tConfig[1]) + SQ(tConfig[3]) + SQ(tConfig[5]));
        // if (diff < 0.1) {
        //   for (int j = 0; j < cdim; j++) {
        //     root->particles[idx][j] = tempConfig[j];
        //   }
        //   idx ++;
        // }
      }
      for (int i = 0; i < n; i ++ ) {
        weight[i] /= sum;
        // cout << weight[i] << endl;
      }
      
      resampleParticles(root->particles, root->particlesPrev, n, weight);
      root->sampleParticles();
      root->propagate();
    } else {
      // double offset = parent[1]->offset;
      // double tol = parent[1]->tol;
      // Node *edge = parent[1]->node;
      // Node *plane = parent[0]->node;
      // double coeffEdge = pow(edge->numParticles, -0.2) * 0.87055/1.2155/1.2155;
      // double coeffPlane = pow(plane->numParticles, -0.2) * 0.87055/1.2155/1.2155;
      // cov = coeff * cov;
    }

  }
}

// bool Node::update(double cur_M[2][3], double Xstd_ob, double R) {
//   cout << "Start updating!" << endl;
//   std::unordered_set<string> bins;
//   std::random_device rd;
//   std::normal_distribution<double> dist(0, 1);
//   std::uniform_real_distribution<double> distribution(0, numParticles);
//   int i = 0;
//   int count = 0;
//   int count2 = 0;
//   int count3 = 0;
//   bool iffar = false;
//   FullParticles b_X = fullParticlesPrev;
//   int idx = 0;
//   fullCspace tempFullState;
//   tempFullState.resize(fulldim);
//   cspace tempState;
//   double D;
//   double cur_inv_M[2][3];
  
//   double unsigned_dist_check = R + 2 * Xstd_ob;
//   double signed_dist_check = 2 * Xstd_ob;

//   //Eigen::Vector3d gradient;
//   Eigen::Vector3d touch_dir;
//   int num_bins = 0;
//   int count_bar = 0;
//   double coeff = pow(4.0 / ((fulldim + 2.0) * numParticles), 2.0/(fulldim + 4.0)) /1.2155/1.2155;
//   // cout << "Coeff: " << coeff << endl;
//   Eigen::MatrixXd H_cov = coeff * full_cov_mat;
//   // cout << "full_cov_mat: " << full_cov_mat << endl;
//   // cout << "cov_mat: " << cov_mat << endl;
//   // cout << "H_cov: " << H_cov << endl;

//   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(H_cov);
//   Eigen::MatrixXd rot = eigenSolver.eigenvectors(); 
//   Eigen::VectorXd scl = eigenSolver.eigenvalues();
//   for (int j = 0; j < fulldim; j++) {
//     scl(j, 0) = sqrt(max2(scl(j, 0),0));
//   }
//   Eigen::VectorXd samples(fulldim, 1);
//   Eigen::VectorXd rot_sample(fulldim, 1);
//   if (type == 1) { // Plane
//     cout << "Start updating Plane!" << endl;
//     while (i < numParticles && i < maxNumParticles) {
//       idx = int(floor(distribution(rd)));

//       for (int j = 0; j < fulldim; j++) {
//         samples(j, 0) = scl(j, 0) * dist(rd);
//       }
//       rot_sample = rot*samples;
//       for (int j = 0; j < fulldim; j++) {
//         /* TODO: use quaternions instead of euler angles */
//         tempFullState[j] = b_X[idx][j] + rot_sample(j, 0);
//       }
//       for (int j = 0; j < cdim; j++) {
//         /* TODO: use quaternions instead of euler angles */
//         tempState[j] = tempFullState[j];
//       }
//       inverseTransform(cur_M, tempState, cur_inv_M);
//       touch_dir << cur_inv_M[1][0], cur_inv_M[1][1], cur_inv_M[1][2];
//       D = abs(cur_inv_M[0][1] - R);
//       // cout << "D: " << D << endl;
      
//       // if (xind >= (dist_transform->num_voxels[0] - 1) || yind >= (dist_transform->num_voxels[1] - 1) || zind >= (dist_transform->num_voxels[2] - 1))
//       //  continue;
          
//       count += 1;
//       // if (sqrt(count) == floor(sqrt(count))) cout << "DDDD " << D << endl;
//       if (D <= signed_dist_check) {
//         // if (sqrt(count) == floor(sqrt(count))) cout << "D " << D << endl;
//         count2 ++;
//         for (int j = 0; j < cdim; j++) {
//           particles[i][j] = tempState[j];
//         }
//         for (int j = 0; j < fulldim; j++) {
//           fullParticles[i][j] = tempFullState[j];
//         }
//         if (checkEmptyBin(&bins, particles[i]) == 1) {
//           num_bins++;
//           // if (i >= N_MIN) {
//           //int numBins = bins.size();
//           numParticles = min2(maxNumParticles, max2(((num_bins - 1) * 2), N_MIN));
//           // }
//         }
//         //double d = testResult(mesh, particles[i], cur_M, R);
//         //if (d > 0.01)
//         //  cout << cur_inv_M[0][0] << "  " << cur_inv_M[0][1] << "  " << cur_inv_M[0][2] << "   " << d << "   " << D << //"   " << gradient << "   " << gradient.dot(touch_dir) << 
//         //       "   " << dist_adjacent[0] << "   " << dist_adjacent[1] << "   " << dist_adjacent[2] << "   " << particles[i][2] << endl;
//         i += 1;
//       }
//     }
//     cout << "End updating Plane!" << endl;
//   } else if (type == 2) { // Edge
//     // for (int i = 0; i < numParticles; i ++) {
//     //   // for (int j = 0; j < cdim; j ++) {
//     //   //   cout << particles[i][j] << " ,";
//     //   // }
//     //   cout << particles[i][0] << " ,";
      
//     // }
//     // cout << endl;
//     // for (int i = 0; i < numParticles; i ++) {
//     //   // for (int j = 0; j < 18; j ++) {
//     //   //   cout << fullParticles[i][j] << " ,";
//     //   // }
//     //   cout << fullParticles[i][0] << " ,";
//     // }
//     //       cout << endl;

//     while (i < numParticles && i < maxNumParticles) {
//       idx = int(floor(distribution(rd)));

//       for (int j = 0; j < fulldim; j++) {
//         samples(j, 0) = scl(j, 0) * dist(rd);
//       }
//       rot_sample = rot*samples;
//       for (int j = 0; j < fulldim; j++) {
//         /* TODO: use quaternions instead of euler angles */
//         tempFullState[j] = b_X[idx][j] + rot_sample(j, 0);
//       }

//       for (int j = 0; j < cdim; j++) {
//         /* TODO: use quaternions instead of euler angles */
//         tempState[j] = tempFullState[j];
//       }
//       Eigen::Vector3d x1, x2, x0, tmp1, tmp2;
//       x1 << tempState[0], tempState[1], tempState[2];
//       x2 << tempState[3], tempState[4], tempState[5];
//       x0 << cur_M[0][0], cur_M[0][1], cur_M[0][2];
//       tmp1 = x1 - x0;
//       tmp2 = x2 - x1;
//       D = (tmp1.squaredNorm() * tmp2.squaredNorm() - pow(tmp1.dot(tmp2),2)) / tmp2.squaredNorm();
//       D = abs(sqrt(D)- R);
//       // cout << "Cur distance: " << D << endl;
//       // cout << "D: " << D << endl;
      
//       // if (xind >= (dist_transform->num_voxels[0] - 1) || yind >= (dist_transform->num_voxels[1] - 1) || zind >= (dist_transform->num_voxels[2] - 1))
//       //  continue;
          
//       count += 1;
//       // if (sqrt(count) == floor(sqrt(count))) cout << "DDDD " << D << endl;
//       if (D <= signed_dist_check) {
//         // if (sqrt(count) == floor(sqrt(count))) cout << "D " << D << endl;
//         count2 ++;
//         for (int j = 0; j < cdim; j++) {
//           particles[i][j] = tempState[j];
//         }
//         for (int j = 0; j < fulldim; j++) {
//           fullParticles[i][j] = tempFullState[j];
//         }
//         if (checkEmptyBin(&bins, particles[i]) == 1) {
//           num_bins++;
//           // if (i >= N_MIN) {
//           //int numBins = bins.size();
//           numParticles = min2(maxNumParticles, max2(((num_bins - 1) * 2), N_MIN));
//           // }
//         }
//         //double d = testResult(mesh, particles[i], cur_M, R);
//         //if (d > 0.01)
//         //  cout << cur_inv_M[0][0] << "  " << cur_inv_M[0][1] << "  " << cur_inv_M[0][2] << "   " << d << "   " << D << //"   " << gradient << "   " << gradient.dot(touch_dir) << 
//         //       "   " << dist_adjacent[0] << "   " << dist_adjacent[1] << "   " << dist_adjacent[2] << "   " << particles[i][2] << endl;
//         i += 1;
//       }
//     }
//     cout << "End updating Edge!" << endl;
//     Node *p = parent[0]->node;
//     if (p->type == 1) {
//       p->update(cur_M, Xstd_ob, R);
//     }
    
//   }
//   cout << "Number of total iterations: " << count << endl;
//   cout << "Number of iterations after unsigned_dist_check: " << count2 << endl;
//   cout << "Number of iterations before safepoint check: " << count3 << endl;
//   cout << "Number of occupied bins: " << num_bins << endl;
//   cout << "Number of particles: " << numParticles << endl;
  
//   particlesPrev = particles;
//   fullParticlesPrev = fullParticles;

//   double* tmpParticles = new double[numParticles * fulldim];
//   for(int i = 0; i < numParticles; ++i) {
//     for (int j = 0; j < fulldim; j ++) {
//       tmpParticles[i * fulldim + j] = fullParticlesPrev[i][j];
//     }
//   }

//   Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(tmpParticles, fulldim, numParticles);
//   Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
//   full_cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
//   mat = Eigen::Map<Eigen::MatrixXd>((double *)particles.data(), cdim, numParticles);
//   mat_centered = mat.colwise() - mat.rowwise().mean();
//   cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));

//   delete[] tmpParticles;

//   cout << "Start to propogate update to root" << endl;
//   // propagate();

//   cout << "End updating!" << endl;
//   return iffar;
// }


// bool Node::update(double cur_M[2][3], double Xstd_ob, double R) {
//   std::unordered_set<string> bins;
//   std::random_device rd;
//   std::normal_distribution<double> dist(0, 1);
//   std::uniform_real_distribution<double> distribution(0, numParticles);
//   int i = 0;
//   int count = 0;
//   int count2 = 0;
//   int count3 = 0;
//   bool iffar = false;
//   Particles b_X = particlesPrev;
//   int idx = 0;
//   cspace tempState;
//   double D;
//   double cur_inv_M[2][3];
  
//   double unsigned_dist_check = R + 2 * Xstd_ob;
//   double signed_dist_check = 2 * Xstd_ob;

//   //Eigen::Vector3d gradient;
//   Eigen::Vector3d touch_dir;
//   int num_bins = 0;
//   int count_bar = 0;
//   double coeff = pow(numParticles, -0.2) * 0.87055/1.2155/1.2155;
//   Eigen::MatrixXd H_cov = coeff * cov_mat;
//   cout << "H_cov: " << H_cov << endl;

//   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(H_cov);
//   Eigen::MatrixXd rot = eigenSolver.eigenvectors(); 
//   Eigen::VectorXd scl = eigenSolver.eigenvalues();

//   for (int j = 0; j < cdim; j++) {
//     scl(j, 0) = sqrt(scl(j, 0));
//   }
//   Eigen::VectorXd samples(cdim, 1);
//   Eigen::VectorXd rot_sample(cdim, 1);
  
//   if (type == 1) { // Plane
//     while (i < numParticles && i < maxNumParticles) {
//       idx = int(floor(distribution(rd)));

//       for (int j = 0; j < cdim; j++) {
//         samples(j, 0) = scl(j, 0) * dist(rd);
//       }
//       rot_sample = rot*samples;
//       for (int j = 0; j < cdim; j++) {
//         /* TODO: use quaternions instead of euler angles */
//         tempState[j] = b_X[idx][j] + rot_sample(j, 0);
//       }

//       inverseTransform(cur_M, tempState, cur_inv_M);
//       touch_dir << cur_inv_M[1][0], cur_inv_M[1][1], cur_inv_M[1][2];
//       D = abs(cur_inv_M[0][1] - R);
//       // cout << "D: " << D << endl;
      
//       // if (xind >= (dist_transform->num_voxels[0] - 1) || yind >= (dist_transform->num_voxels[1] - 1) || zind >= (dist_transform->num_voxels[2] - 1))
//       //  continue;
          
//       count += 1;
//       // if (sqrt(count) == floor(sqrt(count))) cout << "DDDD " << D << endl;
//       if (D <= signed_dist_check) {
//         // if (sqrt(count) == floor(sqrt(count))) cout << "D " << D << endl;
//         count2 ++;
//         for (int j = 0; j < cdim; j++) {
//           particles[i][j] = tempState[j];
//         }

//         if (checkEmptyBin(&bins, particles[i]) == 1) {
//           num_bins++;
//           // if (i >= N_MIN) {
//           //int numBins = bins.size();
//           numParticles = min2(maxNumParticles, max2(((num_bins - 1) * 2), N_MIN));
//           // }
//         }
//         //double d = testResult(mesh, particles[i], cur_M, R);
//         //if (d > 0.01)
//         //  cout << cur_inv_M[0][0] << "  " << cur_inv_M[0][1] << "  " << cur_inv_M[0][2] << "   " << d << "   " << D << //"   " << gradient << "   " << gradient.dot(touch_dir) << 
//         //       "   " << dist_adjacent[0] << "   " << dist_adjacent[1] << "   " << dist_adjacent[2] << "   " << particles[i][2] << endl;
//         i += 1;
//       }
//     }

//   } else if (type == 2) { // Edge
//     while (i < numParticles && i < maxNumParticles) {
//       idx = int(floor(distribution(rd)));

//       for (int j = 0; j < cdim; j++) {
//         samples(j, 0) = scl(j, 0) * dist(rd);
//       }
//       rot_sample = rot*samples;
//       for (int j = 0; j < cdim; j++) {
//         /* TODO: use quaternions instead of euler angles */
//         tempState[j] = b_X[idx][j] + rot_sample(j, 0);
//       }
//       Eigen::Vector3d x1, x2, x0, tmp1, tmp2;
//       x1 << tempState[0], tempState[1], tempState[2];
//       x2 << tempState[3], tempState[4], tempState[5];
//       x0 << cur_M[0][0], cur_M[0][1], cur_M[0][2];
//       tmp1 = x1 - x0;
//       tmp2 = x2 - x1;
//       D = (tmp1.squaredNorm() * tmp2.squaredNorm() - pow(tmp1.dot(tmp2),2)) / tmp2.squaredNorm();
//       D = abs(sqrt(D)- R);
//       // cout << "D: " << D << endl;
      
//       // if (xind >= (dist_transform->num_voxels[0] - 1) || yind >= (dist_transform->num_voxels[1] - 1) || zind >= (dist_transform->num_voxels[2] - 1))
//       //  continue;
          
//       count += 1;
//       // if (sqrt(count) == floor(sqrt(count))) cout << "DDDD " << D << endl;
//       if (D <= signed_dist_check) {
//         // if (sqrt(count) == floor(sqrt(count))) cout << "D " << D << endl;
//         count2 ++;
//         for (int j = 0; j < cdim; j++) {
//           particles[i][j] = tempState[j];
//         }

//         if (checkEmptyBin(&bins, particles[i]) == 1) {
//           num_bins++;
//           // if (i >= N_MIN) {
//           //int numBins = bins.size();
//           numParticles = min2(maxNumParticles, max2(((num_bins - 1) * 2), N_MIN));
//           // }
//         }
//         //double d = testResult(mesh, particles[i], cur_M, R);
//         //if (d > 0.01)
//         //  cout << cur_inv_M[0][0] << "  " << cur_inv_M[0][1] << "  " << cur_inv_M[0][2] << "   " << d << "   " << D << //"   " << gradient << "   " << gradient.dot(touch_dir) << 
//         //       "   " << dist_adjacent[0] << "   " << dist_adjacent[1] << "   " << dist_adjacent[2] << "   " << particles[i][2] << endl;
//         i += 1;
//       }
//     }
//     Node *p = parent[0]->node;
//     if (p->type == 1) {
//       p->update(cur_M, Xstd_ob, R);
//     }
//   }
//   cout << "Number of total iterations: " << count << endl;
//   cout << "Number of iterations after unsigned_dist_check: " << count2 << endl;
//   cout << "Number of iterations before safepoint check: " << count3 << endl;
//   cout << "Number of occupied bins: " << num_bins << endl;
//   cout << "Number of particles: " << numParticles << endl;
  
//   particlesPrev = particles;
//   Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)particlesPrev.data(), cdim, numParticles);
//   Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
//   cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
//   cout << "Start to propogate update to root" << endl;
//   // propagate();


//   return iffar;
// }


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


void transPointConfig(cspace baseConfig, cspace relativeConfig, cspace &absoluteConfig) {
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
  for (int i = 0; i < Node::cdim; i ++) {
    fullConfig[idx + i] = config[i];
  }
}


void copyParticles(cspace config, jointCspace &jointConfig, int idx) {
  for (int i = 0; i < Node::cdim; i ++) {
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
  for (int i = 0; i < Node::cdim; i++) {
    s += floor(config[i] / DISPLACE_INTERVAL);
    s += ":";
  }
  if (set->find(s) == set->end()) {
    set->insert(s);
    return 1;
  }
  return 0;
}