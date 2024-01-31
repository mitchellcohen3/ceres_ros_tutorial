#include "Visualization.h"
#include <std_msgs/ColorRGBA.h>
#include <tf/transform_broadcaster.h>
#include <vector>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud.h>

OdometryViz::OdometryViz(std::string pub_name, ros::NodeHandle &n)
{
    publisher = n.advertise<nav_msgs::Odometry>(pub_name, 1000);
}

void OdometryViz::publish(const Eigen::Matrix3d &attitude, const Eigen::Vector3d &position,
                          double t)
{
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(t);
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";
    odometry.pose.pose.position.x = position.x();
    odometry.pose.pose.position.y = position.y();
    odometry.pose.pose.position.z = position.z();
    Eigen::Quaterniond Q(attitude);
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();
    publisher.publish(odometry);

    tf::StampedTransform stamped_transform = getStampedTransformFromPose(attitude, position, t);
    stamped_transform.frame_id_ = "world";
    stamped_transform.child_frame_id_ = "body";
    broadcaster.sendTransform(stamped_transform);
}

PointCloudViz::PointCloudViz(std::string pub_name, ros::NodeHandle &n) {
  publisher = n.advertise<sensor_msgs::PointCloud>(pub_name, 1000);
}

void PointCloudViz::publish(const std::vector<Eigen::Vector3d> &points) {
  sensor_msgs::PointCloud point_cloud;
  point_cloud.header.frame_id = "world";
  for (auto const &point : points) {
    geometry_msgs::Point32 p;
    p.x = point.x();
    p.y = point.y();
    p.z = point.z();
    point_cloud.points.push_back(p);
  }
  publisher.publish(point_cloud);
}

tf::StampedTransform getStampedTransformFromPose(const Eigen::Matrix3d &attitude,
                                                const Eigen::Vector3d &position, double t)
{
    tf::Transform transform;
    tf::Quaternion q;
    Eigen::Quaterniond quat(attitude);
    q.setW(quat.w());
    q.setX(quat.x());
    q.setY(quat.y());
    q.setZ(quat.z());
    transform.setRotation(q);
    transform.setOrigin(tf::Vector3(position.x(), position.y(), position.z()));

    // Create a stamped transform
    tf::StampedTransform stamped_transform(transform, ros::Time(t), "world", "body");
    return stamped_transform;
}