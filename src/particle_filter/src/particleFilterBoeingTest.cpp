#include <ros/ros.h>
#include "particleFilter.h"
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseArray.h>
#include <std_msgs/Empty.h>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <tf/transform_listener.h>
#include "definitions.h"
#include "particle_filter/PFilterInit.h"
#include "particle_filter/AddObservation.h"
#include "stlParser.h"
// #include <custom_ray_trace/stlParser.h>
// #include <custom_ray_trace/rayTracer.h>
#include <math.h>
#include <string>
#include <array>

// #define POINT_CLOUD
#define NUM_PARTICLES 800
typedef array<array<float, 3>, 4> vec4x3;

#ifdef POINT_CLOUD
pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr1(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr2(new pcl::PointCloud<pcl::PointXYZ>);
bool update;
boost::mutex updateModelMutex;
void visualize();
#endif

class PFilterTest
{
private:
  ros::NodeHandle n;
  ros::Subscriber sub_init;
  ros::Subscriber sub_request_particles;
  ros::ServiceServer srv_add_obs;
  // ros::Publisher pub_particles;

  std::vector<std::string> datum_name_vec;
  std::vector<int> datum_idx_vec;
  std::vector<ros::Publisher> pub_particles_vec;
  ros::Publisher pub_hole;


  tf::StampedTransform trans_;
  // tf::StampedTransform trans1_;
  std::string cadName;
  distanceTransform *dist_transform;
  std::string stlFileDir;
  // ParticleHandler pHandler;

  bool getMesh(std::string filename, vector<vec4x3> &datumMesh);

public:
  vector<vec4x3> mesh;
  int num_voxels[3];
  geometry_msgs::PoseArray getParticlePoseArray(int idx);
  geometry_msgs::PoseArray getHoleParticlePoseArray();
  particleFilter pFilter_;
  PFilterTest(int n_particles, jointCspace b_init[2], std::vector<std::string> datum_name_vec);
  // void addObs(geometry_msgs::Point obs);
  bool addObs(particle_filter::AddObservation::Request &req,
	      particle_filter::AddObservation::Response &resp);
  void sendParticles(std_msgs::Empty);
};

void computeInitialDistribution(jointCspace binit[2], std::vector<std::string> datum_name_vec, ros::NodeHandle n)
{

  std::vector<double> uncertainties;
  std::vector<double> pFrame;
  for (int i = 0; i < datum_name_vec.size(); i ++) {
    if(!n.getParam("/" + datum_name_vec[i] + "/initial_uncertainties", uncertainties)){
      ROS_INFO("Failed to get param");
      uncertainties.resize(6);
    }

    
    if(!n.getParam("/" + datum_name_vec[i] + "/prior", pFrame)){
      ROS_INFO("Failed to get param particle_frame");
      pFrame.resize(6);
    }
    for (int j = 0; j < CDIM; j ++) {
      binit[0][j + i * CDIM] = pFrame[j];
      binit[1][j + i * CDIM] = uncertainties[j];
    }
  }

}

// double SQ(double d)
// {
//   return d*d;
// }


/*
 *  Converts a cspace pose to a tf::Pose
 */
tf::Pose poseAt(cspace particle_pose)
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

void PFilterTest::sendParticles(std_msgs::Empty emptyMsg)
{
  // pub_particles.publish(getParticlePoseArray(0));
  pub_hole.publish(getHoleParticlePoseArray());
  int size = pub_particles_vec.size();
  for (int i = 0; i < size; i ++) {
    pub_particles_vec[i].publish(getParticlePoseArray(datum_idx_vec[i]));
  }
  // pub_particles1.publish(getParticlePoseArray(5));
  // pub_particles2.publish(getParticlePoseArray(6));
}

bool PFilterTest::addObs(particle_filter::AddObservation::Request &req,
			 particle_filter::AddObservation::Response &resp)
{
  geometry_msgs::Point obs = req.p;
  geometry_msgs::Point dir = req.dir;
  int datum = req.datum;
  ROS_INFO("Current update datum: %d", datum);
  ROS_INFO_STREAM("Current update datum name: " << datum_name_vec[datum]);
  ROS_INFO("Adding Observation...");
  ROS_INFO("point: %f, %f, %f", obs.x, obs.y, obs.z);
  ROS_INFO("dir: %f, %f, %f", dir.x, dir.y, dir.z);
  double obs2[2][3] = {{obs.x, obs.y, obs.z}, {dir.x, dir.y, dir.z}};

  vector<vec4x3> datumMesh;
  getMesh(stlFileDir + datum_name_vec[datum] + ".stl", datumMesh);

  // pFilter_.addObservation(obs2, mesh, dist_transform, 0, datum);
  pFilter_.addObservation(obs2, datumMesh, dist_transform, 0, datum);
  ROS_INFO("...Done adding observation");
  // pub_particles.publish(getParticlePoseArray(0));
  int size = pub_particles_vec.size();
  for (int i = 0; i < size; i ++) {
    pub_particles_vec[i].publish(getParticlePoseArray(datum_idx_vec[i]));
  }
  pub_hole.publish(getHoleParticlePoseArray());
  // pub_particles1.publish(getParticlePoseArray(5));
  // pub_particles2.publish(getParticlePoseArray(6));
  return true;
}


bool PFilterTest::getMesh(std::string stlFilePath, vector<vec4x3> &datumMesh){
  datumMesh = importSTL(stlFilePath);
}



geometry_msgs::PoseArray PFilterTest::getHoleParticlePoseArray()
{
  std::vector<cspace> particles;
  pFilter_.getHoleParticles(particles);
  // tf::Transform trans = pHandler.getTransformToPartFrame();
  tf::Transform trans = trans_;
  // cspace particles_est_stat;
  // cspace particles_est;
  // pFilter_.estimateGaussian(particles_est, particles_est_stat);
  geometry_msgs::PoseArray poseArray;
  for(int i=0; i<50; i++){
    tf::Pose pose = poseAt(particles[i]);
    geometry_msgs::Pose pose_msg;
    tf::poseTFToMsg(trans*pose, pose_msg);
    poseArray.poses.push_back(pose_msg);
    // ROS_INFO("Pose %d: %f, %f, %f", i, poseArray.poses[i].position.x,
    //       poseArray.poses[i].position.y, 
    //       poseArray.poses[i].position.z);

  }
  return poseArray;
}

geometry_msgs::PoseArray PFilterTest::getParticlePoseArray(int idx)
{
  std::vector<cspace> particles;
  pFilter_.getAllParticles(particles, idx);
  // tf::Transform trans = pHandler.getTransformToPartFrame();
  tf::Transform trans = trans_;
  // if (idx == 5) trans = trans1_;

  #ifdef POINT_CLOUD
  boost::mutex::scoped_lock updateLock(updateModelMutex);	
  basic_cloud_ptr1->points.clear();
  basic_cloud_ptr2->points.clear();
  int numParticles = pFilter_.getNumParticles();
  for (int j = 0; j < numParticles; j++ ) {
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
  #endif

  // cspace particles_est_stat;
  // cspace particles_est;
  // pFilter_.estimateGaussian(particles_est, particles_est_stat);
  geometry_msgs::PoseArray poseArray;
  for(int i=0; i<50; i++){
    tf::Pose pose = poseAt(particles[i]);
    geometry_msgs::Pose pose_msg;
    tf::poseTFToMsg(trans*pose, pose_msg);
    poseArray.poses.push_back(pose_msg);
    // ROS_INFO("Pose %d: %f, %f, %f", i, poseArray.poses[i].position.x,
    // 	     poseArray.poses[i].position.y, 
    // 	     poseArray.poses[i].position.z);

  }
  return poseArray;
}

#ifdef POINT_CLOUD
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
	viewer->addText("x y z", 15, 120, 20, 1, 1, 1, "v1 text", v1);

	int v2 = 1;
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor (0, 0, 0, v2);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,1,1, "sample cloud2", v2);
	viewer->addText("roll pitch yaw", 15, 120, 20, 1, 1, 1, "v2 text", v2);
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
#endif

PFilterTest::PFilterTest(int n_particles, jointCspace b_init[2], std::vector<std::string> datum_name_vec) :
  pFilter_(n_particles, b_init, 0.0008, 0.000),
  num_voxels{200, 200, 200}//,
  // pFilter_(n_particles, b_init, 0.001, 0.0025, 0.0001, 0.00),
  // num_voxels{300, 300, 300}//,
  //dist_transform(num_voxels)
  // particleFilter (int n_particles, cspace b_init[2], 
  // 				double Xstd_ob=0.0001 (measurement error), 
  //                            double Xstd_tran=0.0025, (gausian kernel sampling std
  // 				double Xstd_scatter=0.0001, (scatter particles a little before computing mean of distance transform
  //                            double R=0.01) (probe radius);



{
  this->datum_name_vec = datum_name_vec;
  if(!n.getParam("localization_object", cadName)){
    ROS_INFO("Failed to get param: localization_object");
  }
  // sub_init = n.subscribe("/particle_filter_init", 1, &PFilterTest::initDistribution, this);
  sub_request_particles = n.subscribe("/request_particles", 1, &PFilterTest::sendParticles, this);
  srv_add_obs = n.advertiseService("/particle_filter_add", &PFilterTest::addObs, this);
  pub_hole = n.advertise<geometry_msgs::PoseArray>("/hole/particles_from_filter", 5);
  for (int i = 0; i < datum_name_vec.size(); i ++) {
    pub_particles_vec.push_back(n.advertise<geometry_msgs::PoseArray>("/" + datum_name_vec[i] + "/particles_from_filter", 5));
    datum_idx_vec.push_back(i);

  }
  
  ROS_INFO("Loading Boeing Particle Filter");

  if(!n.getParam("/localization_object_dir", stlFileDir)){
    ROS_INFO("Failed to get param: stlFileDir");
  }


  std::string stlFilePath;
  if(!n.getParam("/localization_object_filepath", stlFilePath)){
    ROS_INFO("Failed to get param: stlFilePath");
  }

  // getMesh("boeing_part.stl");
  getMesh(stlFilePath, mesh);
  // getMesh("part.stl");
  //int num_voxels[3] = { 200,200,200 };
  //dist_transform(num_voxels);
  ROS_INFO("start create dist_transform");
  dist_transform = new distanceTransform(num_voxels);


  tf::TransformListener tf_listener_;
  // tf_listener_.waitForTransform("/my_frame", "wood_boeing", ros::Time(0), ros::Duration(10.0));
  // tf_listener_.lookupTransform("wood_boeing", "/my_frame", ros::Time(0), trans_);
  tf_listener_.waitForTransform("/my_frame", "right_datum", ros::Time(0), ros::Duration(10.0));
  tf_listener_.lookupTransform("right_datum", "/my_frame", ros::Time(0), trans_);
  ROS_INFO("finish create dist_transform");

  #ifdef POINT_CLOUD
  std::vector<cspace> particles;
  pFilter_.getAllParticles(particles, 0);
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
  #endif
  ros::Duration(1.0).sleep();
  // pub_particles.publish(getParticlePoseArray(0));
  pub_hole.publish(getHoleParticlePoseArray());
  int size = pub_particles_vec.size();
  for (int i = 0; i < size; i ++) {
    pub_particles_vec[i].publish(getParticlePoseArray(datum_idx_vec[i]));
  }
  // pub_particles1.publish(getParticlePoseArray(5));
  // pub_particles2.publish(getParticlePoseArray(6));
}



int main(int argc, char **argv)
{
  #ifdef POINT_CLOUD
  update = false;
  boost::thread workerThread(visualize);
  #endif
  ros::init(argc, argv, "pfilterTest");
  ros::NodeHandle n;
  // ros::Publisher pub = n.advertise<geometry_msgs::PoseArray>("/particles_from_filter", 5);

  ROS_INFO("Testing particle filter");
  
  std::vector<std::string> datum_name_vec;
  if(!n.getParam("/datum_list", datum_name_vec)){
    ROS_INFO("Failed to get param: datum_list");
  }

  jointCspace b_Xprior[2];	
  computeInitialDistribution(b_Xprior, datum_name_vec, n);
  PFilterTest pFilterTest(NUM_PARTICLES, b_Xprior, datum_name_vec);
  
  ros::spin();
  #ifdef POINT_CLOUD
  workerThread.interrupt();
  workerThread.join();
  #endif
}
