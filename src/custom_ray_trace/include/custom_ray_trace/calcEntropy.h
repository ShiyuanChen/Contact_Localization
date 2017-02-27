#ifndef CALC_ENTROPY_H
#define CALC_ENTROPY_H
#include <vector>
#include <map>

struct Bin {
  // std::vector<CalcEntropy::ConfigDist> element;
  std::vector<int> particleIds;
};

namespace CalcEntropy{
  typedef std::vector<int> BinId;
  
  struct ConfigDist {
    double dist;
    int id;
  };

  struct BinWithParticles{
    double binProbability;
    //map from Particle ID to probability of that particle in this bin
    std::map<int, double> particles;
  };

  struct ParticlesWithBin{
    double probability;
    //map from binId to weighted bins this particle is in
    std::map<BinId, double> bin;
  };

  struct ProcessedHistogram{
    std::map<BinId, BinWithParticles> bin;
    std::vector<ParticlesWithBin> particle;
  };
  

  ProcessedHistogram processMeasurements(std::vector<ConfigDist> p, double binSize, int numConfigs);
  ProcessedHistogram combineHist(const ProcessedHistogram &hist1, 
				 const ProcessedHistogram &hist2);
  void getHist(std::vector<ConfigDist> &distances, double binSize, std::vector<Bin> &hist);
  double calcCondDisEntropy(const ProcessedHistogram &procHist);
  double calcCondDisEntropyPerBin(const std::vector<Bin> &hist);
  double calcIG(const std::vector<ConfigDist> &distances, double binSize, int numParticles);
  double calcFullStateIG(double H_Y_given_X, int numParticles);
  double calcIG(const ProcessedHistogram &procHist, int numParticles);

}

#endif


