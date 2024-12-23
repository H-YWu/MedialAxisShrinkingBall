/*
Copyright (c) 2024 Haoyang Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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