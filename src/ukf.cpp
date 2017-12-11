#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(NUM_STATE_DIM);

  // initial covariance matrix
  P_ = MatrixXd(NUM_STATE_DIM, NUM_STATE_DIM);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  /*****************************************************************************
    *  Initialization
    ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state with the first measurement.
      * Create the covariance matrix.
      * */
    // first measurement
    cout << "Initialing EKF" << endl;

    n_aug_ = NUM_STATE_DIM + 2;
    P_ << 0.5, 0,   0, 0,   0,
            0, 0.5, 0, 0,   0,
            0, 0,   1, 0,   0,
            0, 0,   0, 0.5, 0,
            0, 0,   0, 0,   0.5;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double px, py, vx, vy;

      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      px = rho * cos(phi);
      py = rho * sin(phi);


      x_ << px, py, 0, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
      previous_timestamp_ = meas_package.timestamp_;
    } else {
      cout << "Received unknown update type!? : " << meas_package.sensor_type_ << "\n";
    }

    is_initialized_ = true;

    // done initializing, no need to predict or update
    return;
  }


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  //compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;

  Prediction(previous_timestamp_);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else {
    cout << "Received unknown update type!? : " <<  meas_package.sensor_type_ << "\n";
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd sigma_points = generate_sigma_points();
  MatrixXd sigma_predict = predict_sigma_points(sigma_points, delta_t);
  predict_mean_and_covariance(sigma_predict);
}


MatrixXd UKF::generate_sigma_points() {
    //Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a = 0.2;

    //Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd = 0.2;

    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a*std_a;
    P_aug(6,6) = std_yawdd*std_yawdd;

    //create square root matrix
    MatrixXd A = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i+1)       = x_aug + sqrt(LAMBDA + n_aug_) * A.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(LAMBDA + n_aug_) * A.col(i);
    }

    return Xsig_aug;
}

MatrixXd UKF::predict_sigma_points(MatrixXd Xsig_aug, double delta_t) {
    //create matrix with predicted sigma points as columns
    MatrixXd Xsig_pred = MatrixXd(NUM_STATE_DIM, 2 * n_aug_ + 1);

    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i); // row dot
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        double yawd_t = yawd * delta_t;

        //predicted state values
        double px_p, py_p;

        if (fabs(yawd) > 0.001) { //avoid division by zero
            double v_over_yawd = v / yawd;
            px_p = p_x + v_over_yawd * (sin(yaw + yawd_t) - sin(yaw));
            py_p = p_y + v_over_yawd * (cos(yaw) - cos(yaw + yawd_t));
        } else {
            double v_t = v * delta_t;
            px_p = p_x + v_t * cos(yaw);
            py_p = p_y + v_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd_t;
        double yawd_p = yawd;

        // add noise
        double delta_t_2d = delta_t * delta_t * 0.5;
        px_p += delta_t_2d * cos(yaw) * nu_a;
        py_p += delta_t_2d * sin(yaw) * nu_a;
        v_p += delta_t * nu_a;
        yaw_p += delta_t_2d * nu_yawdd;
        yawd_p += delta_t * nu_yawdd;

        //write predicted sigma points into right column
        Xsig_pred(0, i) = px_p;
        Xsig_pred(1, i) = py_p;
        Xsig_pred(2, i) = v_p;
        Xsig_pred(3, i) = yaw_p;
        Xsig_pred(4, i) = yawd_p;
    }

    //cout << "Prediction: " << Xsig_pred << "\n";
    return Xsig_pred;
}

void UKF::predict_mean_and_covariance(MatrixXd Xsig_pred) {

    //create vector for weights
    VectorXd weights = VectorXd(2 * n_aug_ + 1);

    //set weights
    weights(0) = LAMBDA / (LAMBDA + n_aug_);
    for (int i=1; i < weights.rows(); i++) {
        weights(i) = 1 / (2 * (LAMBDA + n_aug_));
    }

    //predict state mean
    for (int i=0; i < weights.rows(); i++) {
        x_ += weights(i) * Xsig_pred.col(i);
    }

    //predict state covariance matrix
    for (int i=0; i < weights.rows(); i++) {
        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x_;
        //cout << "calc'd diff, now for x_diff(3)" << x_diff(3) << " is > M_PI? " << M_PI << endl;

        //angle normalization
        // for some reason: really, really large num_e^15 large num is here, this causes almost infinite loop.
        x_diff(3) = fmod(x_diff(3), 5.); // Need to reconsider initialization.
        //cout << "moded by 10, now for x_diff(3)" << x_diff(3) << " is > M_PI? " << M_PI << endl;
        while (x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
        //cout << "normalized down" << endl;

        while (x_diff(3) < -M_PI) x_diff(3) +=2.*M_PI;
        //cout << "normalized up" << endl;

        P_ += weights(i) * x_diff * x_diff.transpose();
        //cout << "P is incremented" << endl;
    }
    cout << "all done" << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
