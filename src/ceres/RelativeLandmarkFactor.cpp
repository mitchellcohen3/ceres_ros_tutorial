#include "RelativeLandmarkFactor.h"

#include "utils/SE3.h"
#include "utils/SO3.h"

RelativeLandmarkFactorPose::RelativeLandmarkFactorPose(const Eigen::Vector3d &meas_,
                                                       const Eigen::Matrix3d &sqrt_info_,
                                                       const double &stamp_,
                                                       const int &landmark_id_,
                                                       bool print_debug_info_)
    : meas{meas_}, sqrt_info{sqrt_info_}, stamp{stamp_}, landmark_id{landmark_id_},
      print_debug_info{print_debug_info_} {}

bool RelativeLandmarkFactorPose::Evaluate(double const *const *parameters, double *residuals,
                                          double **jacobians) const
{
  // Extract extended pose
  Eigen::Matrix<double, 4, 4> T_ab = SE3::fromCeresParameters(parameters[0]);
  Eigen::Matrix3d C_ab;
  Eigen::Vector3d r_zw_a;
  SE3::toComponents(T_ab, C_ab, r_zw_a);

  // Extract landmark position
  Eigen::Vector3d r_pw_a(parameters[1][0], parameters[1][1], parameters[1][2]);

  // Evaluate measurement model and compute error
  Eigen::Vector3d y_check = C_ab.transpose() * (r_pw_a - r_zw_a);

  Eigen::Vector3d error = meas - y_check;
  Eigen::Map<Eigen::Vector3d> residual(residuals);
  residual = error;
  residual = sqrt_info * residual;

  // Compute the Jacobians
  if (jacobians)
  {
    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 3, 12, Eigen::RowMajor>> jac_pose(jacobians[0]);
      jac_pose.setZero();

      // Compute jacobian with respect to attitude and position
      Eigen::Matrix3d jac_att = -C_ab.transpose() * SO3::cross(r_pw_a);
      Eigen::Matrix3d jac_position = C_ab.transpose();
      jac_pose.leftCols(3) = jac_att;
      jac_pose.rightCols(3) = jac_position;
      jac_pose = sqrt_info * jac_pose;
    }
    if (jacobians[1])
    {
      // Jacobian of residual with respect to landmark parameters
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac_landmark(jacobians[1]);
      jac_landmark.setZero();
      jac_landmark = -C_ab.transpose();
      jac_landmark = sqrt_info * jac_landmark;
    }
  }
  return true;
}