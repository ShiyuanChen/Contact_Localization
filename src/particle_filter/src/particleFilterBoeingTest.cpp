#include <ros/ros.h>
#include "particleFilter.h"
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseArray.h>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include "particle_filter/PFilterInit.h"
#include "particle_filter/AddObservation.h"
#include "stlParser.h"
#include "gazebo_ray_trace/plotRayUtils.h"
#include <math.h>
#include <string>
#include <array>

#define NUM_PARTICLES 500
typedef array<array<float, 3>, 4> vec4x3;
pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr1(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr2(new pcl::PointCloud<pcl::PointXYZ>);
bool update;
boost::mutex updateModelMutex;
void visualize();

class PFilterTest
{
private:
  ros::NodeHandle n;
  ros::Subscriber sub_init;
  ros::ServiceServer srv_add_obs;
  ros::Publisher pub_particles;
  
  distanceTransform *dist_transform;
  PlotRayUtils plt;

  bool getMesh(std::string filename);
public:
  vector<vec4x3> mesh;
  int num_voxels[3];
  geometry_msgs::PoseArray getParticlePoseArray();
  particleFilter pFilter_;
  PFilterTest(int n_particles, particleFilter::cspace b_init[2]);
  // void addObs(geometry_msgs::Point obs);
  bool addObs(particle_filter::AddObservation::Request &req,
	      particle_filter::AddObservation::Response &resp);
};

void computeInitialDistribution(particleFilter::cspace binit[2], ros::NodeHandle n)
{

  std::vector<double> uncertainties;
  if(!n.getParam("/initial_uncertainties", uncertainties)){
    ROS_INFO("Failed to get param");
    uncertainties.resize(6);
  }

  std::vector<double> pFrame;
  if(!n.getParam("/particle_frame", pFrame)){
    ROS_INFO("Failed to get param particle_frame");
    pFrame.resize(6);
  }


  binit[0][0] = pFrame[0] + 0.01;
  binit[0][1] = pFrame[1] + 0.015;
  binit[0][2] = pFrame[2] - 0.015;
  binit[0][3] = pFrame[3] - 0.02;
  binit[0][4] = pFrame[4] + 0.01;
  binit[0][5] = pFrame[5] + 0.03;

  binit[1][0] = uncertainties[0];
  binit[1][1] = uncertainties[1];
  binit[1][2] = uncertainties[2];

  // binit[1][0] = 0.00;
  // binit[1][1] = 0.00;
  // binit[1][2] = 0.00;

  binit[1][3] = uncertainties[3];
  binit[1][4] = uncertainties[4];
  binit[1][5] = uncertainties[5];

  // binit[1][3] = 0;
  // binit[1][4] = 0;
  // binit[1][5] = 0;

}

double SQ(double d)
{
  return d*d;
}


/*
 *  Converts a cspace pose to a tf::Pose
 */
tf::Pose poseAt(particleFilter::cspace particle_pose)
{
  tf::Pose tf_pose;
  tf_pose.setOrigin(tf::Vector3(particle_pose[0], 
				particle_pose[1], 
				particle_pose[2]));
  tf::Quaternion q;
  // q.setRPY(particle_pose[6], particle_pose[5], particle_pose[4]);
  // q.setEulerZYX(particle_pose[6], particle_pose[5], particle_pose[4]);
  q = tf::Quaternion(tf::Vector3(0,0,1), particle_pose[5]) * 
    tf::Quaternion(tf::Vector3(0,1,0), particle_pose[4]) * 
    tf::Quaternion(tf::Vector3(1,0,0), particle_pose[3]);
  tf_pose.setRotation(q);
  
  return tf_pose;

}

bool PFilterTest::addObs(particle_filter::AddObservation::Request &req,
			 particle_filter::AddObservation::Response &resp)
{
  geometry_msgs::Point obs = req.p;
  geometry_msgs::Point dir = req.dir; 
  ROS_INFO("Adding Observation...");
  double obs2[2][3] = {{obs.x, obs.y, obs.z}, {dir.x, dir.y, dir.z}};

  pFilter_.addObservation(obs2, mesh, dist_transform, 0);

  ROS_INFO("...Done adding observation");
  pub_particles.publish(getParticlePoseArray());

}


bool PFilterTest::getMesh(std::string filename){
  std::string localizationObject;
  if(!n.getParam("/localization_object", localizationObject)){
    ROS_INFO("Failed to get param");
  }
  std::string filepath = "/home/shiyuan/Documents/ros_marsarm/src/gazebo_ray_trace/sdf/" + filename;
  if(localizationObject == "boeing_part") {
    
    mesh = importSTL(filepath); 
    return true;
  }
  throw std::invalid_argument("localization object not recognized by particle filter: "
			      + localizationObject);
  return false;
}





geometry_msgs::PoseArray PFilterTest::getParticlePoseArray()
{
  particleFilter::cspace particles[NUM_PARTICLES];
  pFilter_.getAllParticles(particles);
  tf::Transform trans = plt.getTrans();

  boost::mutex::scoped_lock updateLock(updateModelMutex);	
  basic_cloud_ptr1->points.clear();
  basic_cloud_ptr2->points.clear();
  for (int j = 0; j < NUM_PARTICLES; j++ ) {
	pcl::PointXYZ basic_point;
	basic_point.x = particles[j][0] * 2;
	basic_point.y = particles[j][1] * 2;
	basic_point.z = particles[j][2] * 2;
	basic_cloud_ptr1->points.push_back(basic_point);
        basic_point.x = particles[j][3] * 2;
	basic_point.y = particles[j][4] * 2;
	basic_point.z = particles[j][5] * 2;
	basic_cloud_ptr2->points.push_back(basic_point);
  }
  basic_cloud_ptr1->width = (int) basic_cloud_ptr1->points.size ();
  basic_cloud_ptr1->height = 1;
  basic_cloud_ptr2->width = (int) basic_cloud_ptr2->points.size ();
  basic_cloud_ptr2->height = 1;
  update = true;
  updateLock.unlock();


  geometry_msgs::PoseArray poseArray;
  for(int i=0; i<500; i++){
    tf::Pose pose = poseAt(particles[i]);
    geometry_msgs::Pose pose_msg;
    tf::poseTFToMsg(trans*pose, pose_msg);
    poseArray.poses.push_back(pose_msg);
  }
  ROS_INFO("Pose 1: %f%f%f", poseArray.poses[0].position.x,
	   poseArray.poses[0].position.y, 
	   poseArray.poses[0].position.z);
  return poseArray;
}

/*
 * Visualize particles
 */
void visualize()
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->initCameraParameters ();

	int v1 = 0;
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor (0, 0, 0, v1);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,1,1, "sample cloud1", v1);
	//viewer->addCoordinateSystem (1.0,v1);

	int v2 = 1;
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor (0, 0, 0, v2);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,1,1, "sample cloud2", v2);
	viewer->addCoordinateSystem (1.0);
	try {
		while (!viewer->wasStopped ())
		{
			boost::mutex::scoped_lock updateLock(updateModelMutex);
			if(update)
			{
				if(!viewer->updatePointCloud<pcl::PointXYZ>(basic_cloud_ptr1, "sample cloud1"))
					viewer->addPointCloud<pcl::PointXYZ>(basic_cloud_ptr1, "sample cloud1", v1);
				if(!viewer->updatePointCloud<pcl::PointXYZ>(basic_cloud_ptr2, "sample cloud2"))
					viewer->addPointCloud<pcl::PointXYZ>(basic_cloud_ptr2, "sample cloud2", v2);
				update = false;
			}
			updateLock.unlock();
			viewer->spinOnce (100);
		}
	} catch(boost::thread_interrupted&)
	{
		viewer->close();
		return;
	}
}


PFilterTest::PFilterTest(int n_particles, particleFilter::cspace b_init[2]) :
  pFilter_(n_particles, b_init, 0.002, 0.0035, 0.0001, 0.00),
  num_voxels{200, 200, 200}//,
  //dist_transform(num_voxels)
  // particleFilter (int n_particles,
  // 		  double Xstd_ob=0.0001, double Xstd_tran=0.0025,
  // 		  double Xstd_scatter=0.0001, double R=0.0005);
  // particleFilter::particleFilter(int n_particles, cspace b_init[2],
  // 			       double Xstd_ob, double Xstd_tran,
  // 			       double Xstd_scatter, double R)


{
  
  // sub_init = n.subscribe("/particle_filter_init", 1, &PFilterTest::initDistribution, this);
  srv_add_obs = n.advertiseService("/particle_filter_add", &PFilterTest::addObs, this);
  pub_particles = n.advertise<geometry_msgs::PoseArray>("/particles_from_filter", 5);
  ROS_INFO("Testing Boeing");
  getMesh("boeing_part.stl");
  //int num_voxels[3] = { 200,200,200 };
  //dist_transform(num_voxels);
  dist_transform = new distanceTransform(num_voxels);

  particleFilter::cspace particles[NUM_PARTICLES];
  pFilter_.getAllParticles(particles);
  boost::mutex::scoped_lock updateLock(updateModelMutex);	
  basic_cloud_ptr1->points.clear();
  basic_cloud_ptr2->points.clear();
  for (int j = 0; j < NUM_PARTICLES; j++ ) {
	pcl::PointXYZ basic_point;
	basic_point.x = particles[j][0] * 2;
	basic_point.y = particles[j][1] * 2;
	basic_point.z = particles[j][2] * 2;
	basic_cloud_ptr1->points.push_back(basic_point);
        basic_point.x = particles[j][3] * 2;
	basic_point.y = particles[j][4] * 2;
	basic_point.z = particles[j][5] * 2;
	basic_cloud_ptr2->points.push_back(basic_point);
  }
  basic_cloud_ptr1->width = (int) basic_cloud_ptr1->points.size ();
  basic_cloud_ptr1->height = 1;
  basic_cloud_ptr2->width = (int) basic_cloud_ptr2->points.size ();
  basic_cloud_ptr2->height = 1;
  update = true;
  updateLock.unlock();
}

int main(int argc, char **argv)
{ 
  //update = false;
  boost::thread workerThread(visualize);
  ros::init(argc, argv, "pfilterTest");
  ros::NodeHandle n;
  // ros::Publisher pub = n.advertise<geometry_msgs::PoseArray>("/particles_from_filter", 5);

  ROS_INFO("Testing particle filter");
  
  particleFilter::cspace b_Xprior[2];	
  computeInitialDistribution(b_Xprior, n);
  PFilterTest pFilterTest(NUM_PARTICLES, b_Xprior);
  
  ros::spin();
  workerThread.interrupt();
  workerThread.join();

}