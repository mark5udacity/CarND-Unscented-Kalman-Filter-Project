#ifndef UKF_H
#define UKF_H

static const int NUM_RADAR_DIM = 3;

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
private:
    MatrixXd generate_sigma_points();

    MatrixXd predict_sigma_points(MatrixXd Xsig_aug, double delta_t);

    void predict_mean_and_covariance(MatrixXd Xsig_pred);

    MatrixXd xIdentity;
    MatrixXd R_;
    MatrixXd H_;
    MatrixXd Ht;
    VectorXd weights;
    MatrixXd R_radar_;
    Tools tools;

public:

    ///* initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    // previous timestamp
    long long previous_timestamp_;

    ///* if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    ///* if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    VectorXd x_;

    ///* state covariance matrix
    MatrixXd P_;

    ///* predicted sigma points matrix
    MatrixXd Xsig_pred_;

    ///* time when the state is true, in us
    long long time_us_;

    ///* Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    ///* Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    ///* Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    ///* Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    ///* Radar measurement noise standard deviation radius in m
    double std_radr_;

    ///* Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    ///* Radar measurement noise standard deviation radius change in m/s
    double std_radrd_;

    ///* Weights of sigma points
    VectorXd weights_;

    ///* State dimension
    int n_x_;

    ///* Augmented state dimension
    int n_aug_;

    int n_sig_;

    ///* Sigma point spreading parameter
    double lambda_;


    /**
     * Constructor
     */
    UKF();

    /**
     * Destructor
     */
    virtual ~UKF();

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param delta_t Time between k and k+1 in s
     */
    MatrixXd Prediction(double delta_t);

    /**
     * Updates the state and the state covariance matrix using a laser measurement
     * @param rawMeasurement The measurement at k+1
     */
    void UpdateLidar(VectorXd rawMeasurement, MatrixXd matrix);

    /**
     * Updates the state and the state covariance matrix using a radar measurement
     * @param raw_measurement The measurement at k+1
     */
    void UpdateRadar(VectorXd raw_measurement, MatrixXd matrix);
};

#endif /* UKF_H */
