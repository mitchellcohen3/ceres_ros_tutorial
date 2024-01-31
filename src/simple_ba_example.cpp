#include <Eigen/Dense>
#include <algorithm>
#include <ceres/ceres.h>
#include <iomanip>
#include <random>
#include "ceres/PoseLocalParameterization.h"
#include "ceres/RelativeLandmarkFactor.h"
#include "utils/Visualization.h"
#include "utils/SE3.h"
#include "utils/SO3.h"

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

ceres::Solver::Options createCeresOptions()
{
  ceres::Solver::Options options;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_solver_time_in_seconds = 20;
  options.max_num_iterations = 500;
  options.minimizer_progress_to_stdout = false;
  options.function_tolerance = 1e-8;
  options.gradient_tolerance = 1e-8;
  options.parameter_tolerance = 1e-8;

  return options;
}

struct NormalRandomVariable
{
  NormalRandomVariable(Eigen::MatrixXd const &covar)
      : NormalRandomVariable(Eigen::VectorXd::Zero(covar.rows()), covar) {}

  NormalRandomVariable(Eigen::VectorXd const &mean, Eigen::MatrixXd const &covar) : mean(mean)
  {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
    transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

  Eigen::VectorXd mean;
  Eigen::MatrixXd transform;

  Eigen::VectorXd operator()() const
  {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<> dist;

    return mean +
           transform * Eigen::VectorXd{mean.size()}.unaryExpr([&](auto x)
                                                              { return dist(gen); });
  }
};

// Generate poses for the bundle adjustment simulation
std::vector<Eigen::Matrix4d> generatePoses()
{
  std::vector<Eigen::Matrix4d> poses;
  Eigen::Matrix<double, 6, 1> xi_1;
  Eigen::Matrix<double, 6, 1> xi_2;
  Eigen::Matrix<double, 6, 1> xi_3;
  xi_1 << (M_PI / 36.0), 0.0, (M_PI / 12.0), 0.2, -0.5, 0.2;
  xi_2 << (M_PI / 36.0), 0.0, (M_PI / 12.0), -0.4, 2.5, 0.25;
  xi_3 << (M_PI / 12.0), 0.0, (M_PI / 12.0), 0.3, 4.4, -0.3;
  poses.push_back(SE3::expMap(xi_1));
  poses.push_back(SE3::expMap(xi_2));
  poses.push_back(SE3::expMap(xi_3));
  return poses;
}

// Generate landmarks for the bundle adjustment simulation
std::vector<Eigen::Vector3d> generateLandmarks(int num_landmarks_width, int num_landmarks_height, double depth_landmarks)
{
  std::vector<Eigen::Vector3d> landmarks;
  Eigen::VectorXd widths = Eigen::VectorXd::LinSpaced(num_landmarks_width, 0.0, 5.0);
  Eigen::VectorXd heights = Eigen::VectorXd::LinSpaced(num_landmarks_height, 0.0, 5.0);

  int landmark_idx = 0;
  for (int i = 0; i < num_landmarks_width; i++)
  {
    double cur_width = widths(i);
    for (int j = 0; j < num_landmarks_height; j++)
    {
      double cur_height = heights(j);
      Eigen::Vector3d landmark;
      landmark << depth_landmarks, cur_width, cur_height;
      landmarks.push_back(landmark);
    }
  }
  return landmarks;
}

// Compute the Cholesky decomposition of an inverse covariance matrix
Eigen::MatrixXd computeSquareRootInformation(const Eigen::MatrixXd &covariance) {
  // Check that we have a valid covariance that we can get the information of
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(covariance.rows(), covariance.rows());
  Eigen::MatrixXd information = covariance.llt().solve(I);
  if (std::isnan(information.norm())) {
    std::cerr << "Covariance : " << std::endl << covariance << std::endl << std::endl;
    std::cerr << "Inverse covariance : " << std::endl
              << covariance.inverse() << std::endl
              << std::endl;
  }
  // Compute square-root of the information matrix
  Eigen::LLT<Eigen::MatrixXd> lltOfI(information);
  Eigen::MatrixXd sqrt_info = lltOfI.matrixL().transpose();
  return sqrt_info;
}

int main(int argc, char **argv)
{
  // Landmark spacing parameters
  int num_landmarks_width = 20;
  int num_landmarks_height = 20;
  double depth_landmarks = 2;
  // Measurement covariance
  double cov_meas = 0.001;
  // Initialization parameters (used to perturb groundtruth for initial guess)
  double sigma_init_phi = 0.4;
  double sigma_init_r = 0.4;
  double sigma_init_m = 0.1;

  std::vector<Eigen::Matrix4d> poses_gt = generatePoses();
  std::vector<Eigen::Vector3d> landmarks_gt = generateLandmarks(num_landmarks_width, num_landmarks_height, depth_landmarks);
  ROS_INFO_STREAM("Number of poses generated: " << poses_gt.size());
  ROS_INFO_STREAM("Number of landmarks generated: " << landmarks_gt.size());

  // Generate measurements
  Eigen::Matrix3d R_k = cov_meas * Eigen::Matrix3d::Identity();
  NormalRandomVariable noise{R_k};
  std::vector<std::vector<Eigen::Vector3d>> meas_vec;
  for (int pose_idx = 0; pose_idx < poses_gt.size(); pose_idx++)
  {
    std::vector<Eigen::Vector3d> meas_per_pose;
    for (int landmark_idx = 0; landmark_idx < landmarks_gt.size(); landmark_idx++)
    {
      Eigen::Matrix4d cur_pose = poses_gt[pose_idx];
      Eigen::Vector3d cur_landmark = landmarks_gt[landmark_idx];
      // Get landmark resolved in body frame
      Eigen::Matrix3d attitude = cur_pose.block<3, 3>(0, 0);
      Eigen::Vector3d position = cur_pose.block<3, 1>(0, 3);
      Eigen::Vector3d noiseless_meas = attitude.transpose() * (cur_landmark - position);
      Eigen::Vector3d noise_vec = noise();
      Eigen::Vector3d meas = noiseless_meas + noise_vec;
      meas_per_pose.push_back(meas);
    }
    meas_vec.push_back(meas_per_pose);
  }

  // Generate initial estimate by perturbing ground truth
  std::vector<Eigen::Matrix4d> init_poses;
  Eigen::Matrix<double, 6, 6> cov_pose;
  cov_pose.setIdentity();
  cov_pose.block<3, 3>(0, 0) = sigma_init_phi * sigma_init_phi * Eigen::Matrix3d::Identity();
  cov_pose.block<3, 3>(3, 3) = sigma_init_r * sigma_init_r * Eigen::Matrix3d::Identity();
  NormalRandomVariable noise_pose{cov_pose};

  // Initialize first pose to groundtruth
  Eigen::Matrix4d init_est_pose = poses_gt[0];
  init_poses.push_back(init_est_pose);
  for (int i = 1; i < poses_gt.size(); i++)
  {
    Eigen::VectorXd delta_xi = noise_pose();
    Eigen::Matrix4d perturbed_pose = poses_gt[i] * SE3::expMap(delta_xi);
    init_poses.push_back(perturbed_pose);
  }
  // Perturb landmark locations
  std::vector<Eigen::Vector3d> init_landmarks;
  Eigen::Matrix3d cov_landmarks = sigma_init_m * sigma_init_m * Eigen::Matrix3d::Identity();
  NormalRandomVariable noise_landmark{cov_landmarks};
  for (int i = 0; i < landmarks_gt.size(); i++)
  {
    Eigen::VectorXd delta_xi = noise_landmark();
    Eigen::Vector3d estimated_landmark = landmarks_gt[i] + delta_xi;
    init_landmarks.push_back(estimated_landmark);
  }

  // Visualize initial guess
  ros::init(argc, argv, "visualization");
  ros::NodeHandle n("~");
  OdometryViz odom_viz("odometry", n);
  PointCloudViz point_cloud_viz("point_cloud", n);
  ROS_INFO_STREAM("Publishing initial guess over RViz...");
  for (int time_idx = 0; time_idx < 2; time_idx++)
  {
    for (int i = 0; i < init_poses.size(); i++)
    {
      Eigen::Matrix4d pose = init_poses[i];
      Eigen::Matrix3d attitude = pose.block<3, 3>(0, 0);
      Eigen::Vector3d position = pose.block<3, 1>(0, 3);
      odom_viz.publish(attitude, position, 0.0);
    }

    point_cloud_viz.publish(init_landmarks);
    ros::Duration(2.0).sleep();
  }

  // Initialize variable holders needed for Ceres
  int num_poses = init_poses.size();
  int num_landmarks = init_landmarks.size();
  int size_pose = 12;
  int size_landmark = 3;
  double para_pose[num_poses][size_pose];
  double para_landmark[num_landmarks][size_landmark];

  for (int i = 0; i < num_poses; i++)
  {
    Eigen::Matrix4d cur_pose = init_poses[i];
    Eigen::Matrix3d cur_att = cur_pose.block<3, 3>(0, 0);
    Eigen::Vector3d cur_position = cur_pose.block<3, 1>(0, 3);
    // Get the flattened DCM
    Eigen::Matrix<double, 9, 1> attitude = SO3::flatten(cur_att);
    for (int j = 0; j < 9; j++)
    {
      para_pose[i][j] = attitude(j, 0);
    }

    para_pose[i][9] = cur_position.x();
    para_pose[i][10] = cur_position.y();
    para_pose[i][11] = cur_position.z();
  }
  for (int i = 0; i < num_landmarks; i++)
  {
    Eigen::Vector3d cur_landmark = init_landmarks[i];
    para_landmark[i][0] = cur_landmark.x();
    para_landmark[i][1] = cur_landmark.y();
    para_landmark[i][2] = cur_landmark.z();
  }

  // Build Ceres problem
  ceres::Problem problem;
  for (int i = 0; i < num_poses; i++)
  {
    ceres::LocalParameterization *pose_local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(para_pose[i], size_pose, pose_local_parameterization);
  }

  // Set the first pose constant
  problem.SetParameterBlockConstant(para_pose[0]);
  for (int i = 0; i < num_landmarks; i++)
  {
    problem.AddParameterBlock(para_landmark[i], size_landmark);
  }

  // Add one factor for each measurement
  Eigen::Matrix3d sqrt_info = computeSquareRootInformation(R_k);
  for (int pose_idx = 0; pose_idx < meas_vec.size(); pose_idx++)
  {
    std::vector<Eigen::Vector3d> cur_meas_vec = meas_vec[pose_idx];
    for (int landmark_idx = 0; landmark_idx < cur_meas_vec.size(); landmark_idx++)
    {
      Eigen::Vector3d cur_meas = cur_meas_vec[landmark_idx];
      RelativeLandmarkFactorPose *landmark_factor =
          new RelativeLandmarkFactorPose(cur_meas, sqrt_info, 0.0, landmark_idx);
      problem.AddResidualBlock(landmark_factor, NULL, para_pose[pose_idx],
                               para_landmark[landmark_idx]);
    }
  }

  // Solve with Ceres
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 20;
  options.max_solver_time_in_seconds = 50;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  // Get optimized values from solver and display over RViz
  std::vector<Eigen::Matrix4d> optimized_poses;
  for (int i = 0; i < num_poses; i++)
  {
    optimized_poses.push_back(SE3::fromCeresParameters(para_pose[i]));
  }

  std::vector<Eigen::Vector3d> optimized_landmarks;
  for (int i = 0; i < num_landmarks; i++)
  {
    Eigen::Vector3d optimized_landmark(para_landmark[i][0], para_landmark[i][1],
                                       para_landmark[i][2]);
    optimized_landmarks.push_back(optimized_landmark);
  }

  ROS_INFO_STREAM("Publishing optimized values over RViz...")
  for (int i = 0; i < optimized_poses.size(); i++)
  {
    Eigen::Matrix3d attitude = optimized_poses[i].block<3, 3>(0, 0);
    Eigen::Vector3d position = optimized_poses[i].block<3, 1>(0, 3);
    odom_viz.publish(attitude, position, 0.0);
  }

  point_cloud_viz.publish(optimized_landmarks);
  ros::Duration(1.0).sleep();
}