#pragma once

#include <Eigen/Dense>
#include <cmath>

class SO3 {
public:
  static Eigen::Matrix3d cross(const Eigen::Vector3d &x);
  static Eigen::Vector3d vee(const Eigen::Matrix3d &element_so3);
  static Eigen::Matrix3d expMap(const Eigen::Vector3d &phi);
  static Eigen::Vector3d logMap(const Eigen::Matrix3d &element_so3);
  static Eigen::Matrix3d leftJacobian(const Eigen::Vector3d &phi);
  static Eigen::Matrix3d computeJRight(const Eigen::Vector3d &phi);
  static Eigen::Matrix3d leftJacobianInv(const Eigen::Vector3d &phi);
  static Eigen::Matrix3d computeJRightInv(const Eigen::Vector3d &phi);
  static Eigen::Matrix<double, 9, 1> flatten(const Eigen::Matrix3d C);
  static Eigen::Matrix3d unflatten(const Eigen::Matrix<double, 9, 1> vec_C);
  static Eigen::Vector3d toEuler(const Eigen::Matrix3d &C);
};