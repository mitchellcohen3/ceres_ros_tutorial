#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>

class PoseLocalParameterization : public ceres::LocalParameterization
{
public:
  /**
   * @brief State update funciton for the extended Pose state.
   */
  bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
  /**
   * @brief Computes the jacobian in respect to the local parameterization
   *
   * This essentially "tricks" ceres.
   * Instead of doing what ceres wants:
   * dr/dlocal= dr/dglobal * dglobal/dlocal
   *
   * We instead directly do:
   * dr/dlocal= [ dr/dlocal, 0] * [I; 0]= dr/dlocal.
   * Therefore we here define dglobal/dlocal= [I; 0]
   *
   * For example, lets consider orientation parameterized as a 4 DoF quaternion
   * (denoted q). Consdier a residual, written e. In the cost function, the user
   * supplies the Jacobian of the residual wrt the 4 DoF quaternion -> de / dq.
   * Here, the user supplies the Jacobian of the quaternion w.r.t the Lie
   * algebra element xi, -> dq / dxi Interally, Ceres then computes de / dxi =
   * de / dq * dq / dxi. However, when we analytically derive Jacobians, it's
   * often easiest to directly find de / dxi, the Jacobian of the residual w.r.t
   * the Lie algebra. For this, a "trick" can be used where the user sets the
   * ComputeJacobian method in the LocalParameteriation to identity, and
   * supplies the Jacobian de / dxi directly in the CostFunction, wih zero
   * columns in the correct places. For example, see
   *      https://github.com/ceres-solver/ceres-solver/issues/387
   *      https://github.com/ceres-solver/ceres-solver/issues/303
   */
  bool ComputeJacobian(const double *x, double *jacobian) const;
  int GlobalSize() const { return 12; };
  int LocalSize() const { return 6; };

  /**
   * @brief returns the Eigen Jacobian with respect to the local
   * parameterization. Usefor for debugging. Usefor for debugggining
   */
  Eigen::Matrix<double, 12, 6> getEigenJacobian() const;
};
