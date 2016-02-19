/**
 *  Plots a ray and the intersections of that ray with obstacles 
 */

#include "ros/ros.h"

#include "calcEntropy.h"
#include "plotRayUtils.h"
#include <sstream>



int main(int argc, char **argv){

  ros::init(argc, argv, "ray_trace_grid");

  PlotRayUtils plt;
  for(int i = -2; i < 3; i++){
    for(int j = -2; j < 3; j++){
      double x = 1 + 0.2 * (i);
      double y = 2 + 0.2 * (j);
      tf::Point start(x, y, 3.3);
      tf::Point end(x, y, 2.0);



      // plt.plotRay(start, end, false);

 
      // std::vector<double> dist = plt.getDistToParticles(start, end);

      // plt.plotIntersections(start, end, false);
      // double entropy = CalcEntropy::calcDifferentialEntropy(dist);


      // std::stringstream s;
      // std::string entropy_string;
      // // std::sprintf(entropy_string, "%f", entropy);
      // s  << std::fixed << std::setprecision(2) << entropy;
      // plt.labelRay(start, s.str());
      plt.plotCylinder(start, end, 0.01, 0.002);
      ros::Duration(0.2).sleep();
    } 
  }


  return 0;
}