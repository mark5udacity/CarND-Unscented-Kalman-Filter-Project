#include "ukf.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    n_x_ = 5;
    lambda_ = 3 - n_x_;

    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(n_x_);
    xIdentity = MatrixXd::Identity(n_x_, n_x_);

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = M_PI / 5;

    ////////////////// DO NOT MODIFY ////////////////////////////////////////////////////
    // measurement noise values below these are provided by the sensor manufacturer.   //
    // Laser measurement noise standard deviation position1 in m                       //
    /////////////////////////////////////////////////////////////////////////////////////
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03; // in lectures: ? 0.0175;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3; // in lecture?  0.1 ;

    //^^^ DO NOT MODIFY ^^ measurement noise values above ^^these are provided by the sensor manufacturer. ^^

    R_ = MatrixXd(2, 2);
    R_ << 0.0225, 0,
            0, 0.0225;

    H_ = MatrixXd(2, 5);
    H_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;
    Ht = H_.transpose();

    n_aug_ = n_x_ + 2;
    n_sig_ = 2 * n_aug_ + 1;
    
    weights = VectorXd(n_sig_);
    weights.fill(1 / (2 * (lambda_ + n_aug_)));
    weights(0) = lambda_ / (lambda_ + n_aug_);

    R_radar_ = MatrixXd(NUM_RADAR_DIM, NUM_RADAR_DIM);
    R_radar_ << std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;
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
        cout << "Initialing EKF" << endl;

        P_ << 0.5, 0, 0, 0, 0,
                0, 0.5, 0, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 0.5, 0,
                0, 0, 0, 0, 0.5;

        previous_timestamp_ = meas_package.timestamp_;

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            double px, py;

            double rho = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];
            px = rho * cos(phi);
            py = rho * sin(phi);

            x_ << px, py, 0.01, 0.01, 0.01;
        } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.01, 0.01, 0.01;
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
    float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;    //dt - expressed in seconds
    previous_timestamp_ = meas_package.timestamp_;

    // cout << "Change in time: " << dt << endl;
    MatrixXd sigma_points = Prediction(dt);

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
       * Use the sensor type to perform the update step.
       * Update the state and covariance matrices.
     */

    // print the output
    cout << "predicted x_ = " << x_ << endl;
    cout << "predicted P_ = " << P_ << endl;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package.raw_measurements_, sigma_points);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package.raw_measurements_, sigma_points);
    } else {
        cout << "Received unknown update type!? : " << meas_package.sensor_type_ << "\n";
    }

    cout << "post measurement update x_ = " << x_ << endl;
    cout << "post measurement update P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
MatrixXd UKF::Prediction(double delta_t) {
    MatrixXd sigma_points = generate_sigma_points();
    //cout << "Generated sigma points: " << sigma_points << endl;
    MatrixXd sigma_predict = predict_sigma_points(sigma_points, delta_t);
    //cout << "predicted sigm: " << sigma_predict << endl;
    predict_mean_and_covariance(sigma_predict);

    return sigma_predict;
}


MatrixXd UKF::generate_sigma_points() {

    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);

    //create augmented mean state
    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0;
    x_aug(n_x_ + 1) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(P_.rows(), P_.cols()) << P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd A = P_aug.llt().matrixL();

    double sqrtLambda = sqrt(lambda_ + n_aug_);

    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i + 1) = x_aug + sqrtLambda * A.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrtLambda * A.col(i);
    }

    return Xsig_aug;
}

MatrixXd UKF::predict_sigma_points(MatrixXd Xsig_aug, double delta_t) {
    //create matrix with predicted sigma points as columns
    MatrixXd Xsig_pred = MatrixXd(n_x_, n_sig_);

    //predict sigma points
    for (int i = 0; i < n_sig_; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i); // row dot
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

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
    //predict state mean
    x_ = Xsig_pred * weights;

    //predict state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < weights.rows(); i++) {
        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x_;
        //cout << "calc'd diff, now for x_diff(3)" << x_diff(3) << " is > M_PI? " << M_PI << endl;

        //angle normalization
        // for some reason: really, really large num_e^15 large num is here, this causes almost infinite loop.
        //x_diff(3) = fmod(x_diff(3), 5.); // Need to reconsider initialization.
        //cout << "moded by 10, now for x_diff(3)" << x_diff(3) << " is > M_PI? " << M_PI << endl;
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        //cout << "normalized down" << endl;

        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
        //cout << "normalized up" << endl;

        P_ += weights(i) * x_diff * x_diff.transpose();
        //cout << "P is incremented" << endl;
    }
    //cout << "all done" << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(VectorXd rawMeasurement, MatrixXd matrix) {
    /**
    Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    TODO: You'll also need to calculate the lidar NIS.
    */

    // laser
    VectorXd z_pred = H_ * x_;
    VectorXd y = rawMeasurement - z_pred;
    MatrixXd PHt = P_ * Ht;
    MatrixXd S = H_ * PHt + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    P_ = (xIdentity - K * H_) * P_; // alternative: K * H_ * P_
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(VectorXd raw_measurement, MatrixXd Xsig_pred) {
    /**
    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    TODO: You'll also need to calculate the radar NIS.
    */

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(NUM_RADAR_DIM, n_sig_);

    //mean predicted measurement
    VectorXd z_pred = VectorXd(NUM_RADAR_DIM);

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(NUM_RADAR_DIM, NUM_RADAR_DIM);

/*******************************************************************************
 * Where should S go? ^^
 ******************************************************************************/

    //transform sigma points into measurement space
    for (int i = 0; i < weights.rows(); i++) {
        double px = Xsig_pred(0, i);
        double py = Xsig_pred(1, i);
        double pv = Xsig_pred(2, i);
        double yaw = Xsig_pred(3, i);

        double px2 = px * px;
        double py2 = py * py;
        Zsig(0, i) = sqrt(px2 + py2);
        Zsig(1, i) = atan2(py, px);
        Zsig(2, i) = (px * cos(yaw) * pv + py * sin(yaw) * pv);

        double sqrtPxPy = sqrt(px2 + py2);
        if (fabs(sqrtPxPy) > 0.001) {
            Zsig(2, i) /= sqrtPxPy;
        } else {
            cout << "Avoided Divide by zero!!!" << endl;
        }
    }

    //calculate mean predicted measurement
    z_pred.fill(0.0);
    for (int i = 0; i < weights.rows(); i++) {
        z_pred += weights(i) * Zsig.col(i);
    }

    //calculate measurement covariance matrix S
    S.fill(0.0);
    for (int i = 0; i < weights.rows(); i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S += weights(i) * z_diff * z_diff.transpose();
    }

    S += R_radar_;

    //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
    //std::cout << "S: " << std::endl << S << std::endl;


    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, NUM_RADAR_DIM);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x_;
        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = raw_measurement - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    //update state mean and covariance matrix
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
}
