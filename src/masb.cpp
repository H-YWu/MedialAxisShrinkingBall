#include "compute_medial_axis.hpp"
#include "io.hpp"

#include "cnpy.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_dir_of_point_cloud>" << std::endl;
        return 1;
    }
    std::string dir = argv[1];

    std::cout << "Load oriented point cloud from .npy files ...\n";
    Eigen::MatrixXd points = load_npy_to_eigen_matrix(dir + "/coords.npy");
    Eigen::MatrixXd normals = load_npy_to_eigen_matrix(dir + "/normals.npy");

    std::cout << "Compute inside medial axis ...\n";
    Eigen::MatrixXd ma_points_in;
    Eigen::VectorXd ma_radii_in;
    compute_medial_axis(ma_points_in, ma_radii_in, points, normals);
    std::cout << "Compute outside medial axis ...\n";
    Eigen::MatrixXd ma_points_out;
    Eigen::VectorXd ma_radii_out;
    compute_medial_axis(ma_points_out, ma_radii_out, points, -normals);

    save_eigen_matrix_to_xyz(ma_points_in, dir + "/medial_axis_in.xyz");
    save_eigen_matrix_to_xyz(ma_points_out, dir + "/medial_axis_out.xyz");

    std::cout << "Medial axis points saved to 'medial_axis_in.xyz' and 'medial_axis_out.xyz'." << std::endl;

    return 0;
}
