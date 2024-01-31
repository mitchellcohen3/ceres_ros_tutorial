#pragma once

#include <tf/transform_broadcaster.h>
#include <ros/ros.h>
#include <Eigen/Dense>

class OdometryViz
{
public:
    OdometryViz(std::string pub_name, ros::NodeHandle &n);

    void publish(const Eigen::Matrix3d &attitude, const Eigen::Vector3d &position, double t);

    ros::Publisher publisher;
    tf::TransformBroadcaster broadcaster;
};

class PointCloudViz
{
public:
    PointCloudViz(std::string pub_name, ros::NodeHandle &n);

    void publish(const std::vector<Eigen::Vector3d> &points);
    ros::Publisher publisher;
};

tf::StampedTransform getStampedTransformFromPose(const Eigen::Matrix3d &attitude,
                                                 const Eigen::Vector3d &position, double t);