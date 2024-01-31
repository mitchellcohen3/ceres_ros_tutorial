#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>

class RelativeLandmarkFactorPose : public ceres::SizedCostFunction<3, 12, 3>
{
public:
  Eigen::Vector3d meas;
  Eigen::Matrix3d sqrt_info;
  double stamp;
  int landmark_id;
  bool print_debug_info;

  RelativeLandmarkFactorPose(const Eigen::Vector3d &meas, const Eigen::Matrix3d &sqrt_info_,
                             const double &stamp_, const int &landmark_id_,
                             bool print_debug_info_ = false);

  /**
   * @brief Residual and Jacobian computation
   */
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
};