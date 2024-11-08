#ifndef MASB_IO_
#define MASB_IO_

#include "cnpy.h"

#include <Eigen/Dense>

#include <fstream>

void save_eigen_matrix_to_xyz(const Eigen::MatrixXd& emat, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < emat.rows(); i ++) {
        file << emat.row(i);
        file << std::endl;
    }
}

Eigen::MatrixXd load_npy_to_eigen_matrix(const std::string& filename) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    double* data = arr.data<double>();

    Eigen::MatrixXd emat(arr.shape[0], arr.shape[1]);
    emat.setZero();
    for (size_t i = 0; i < arr.shape[0]; ++i) {
        for (size_t j = 0; j < arr.shape[1]; ++j) {
            emat(i, j) = data[i * arr.shape[1] + j];
        }
    }

    return emat;
}

#endif