/*
Copyright (c) 2024 Haoyang Wu 
Copyright (c) 2016 Ravi Peters

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

#ifndef MASB_COMPUTE_MEDIAL_AXIS
#define MASB_COMPUTE_MEDIAL_AXIS

#include <nanoflann.hpp>
#include <Eigen/Dense>

#include <vector>
#include <random>

#include <omp.h>

typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> KDTree;

double compute_radius(const Eigen::VectorXd& p, const Eigen::VectorXd& n, const Eigen::VectorXd& q) {
    double d = (p - q).norm();
    double cos_theta = n.dot(p - q) / d;
    return d / (2 * cos_theta);
}

double cos_angle(const Eigen::VectorXd& p, const Eigen::VectorXd& q) {
   double result = p.dot(q) / (p.norm() * q.norm());
   if (result > 1) return 1;
   else if (result < -1) return -1;
   return result;
}

std::pair<Eigen::VectorXd, double>
compute_single_ma_point(
    const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals,
    KDTree& kd_tree, int i,
    double init_r, double neib_dist,
    int max_iters, double eps, double ndot_thd
) {
    Eigen::VectorXd p = points.row(i).transpose();
    Eigen::VectorXd n = normals.row(i).transpose();

    double r = init_r;
    unsigned int j = 0;
    Eigen::VectorXd c = p - r * n;
    Eigen::VectorXd q = p;
    int qidx = -1;

    while (j < max_iters) {

        // Nearest points to c
        const int k = 10;
        std::vector<size_t> indices(k);
        std::vector<double> dists(k);
        nanoflann::KNNResultSet<double> resultSet(k);
        resultSet.init(&indices[0], &dists[0]);
        kd_tree.index->findNeighbors(resultSet, &c[0], nanoflann::SearchParams(10));

        // Do not shrink
        if (abs(dists[0]-r) < eps) break;

        // Find a suitable q
        double min_ndot = ndot_thd;
        double max_dist_pq = 0;
        qidx = -1;
        for (size_t z = 0; z < k; ++z) {
            size_t idx = indices[z];
            if (idx == i) continue;
            double dist_pq = (p - points.row(idx).transpose()).norm();
            double dist_q = dists[z];
            if (dist_q - dists[0] > eps) continue;
            Eigen::VectorXd qn = normals.row(idx); double ndot = n.dot(qn);
            if ((ndot < min_ndot) || (dist_pq > k * k * neib_dist)) {
                if (dist_pq > max_dist_pq) {
                    max_dist_pq = dist_pq;
                    min_ndot = ndot;
                    qidx = idx;
                }
            }
        }
        if (qidx == -1) break;

        q = points.row(qidx).transpose();
        double r_nxt = compute_radius(p, n, q);
        if (abs(r - r_nxt) < eps) break;

        r = r_nxt;
        c = p - r * n;
        ++j;
    }

    if (j > 0) return {c, r};
    return {c, -1.0};
}

double expected_radius(
    const Eigen::VectorXd& p, const Eigen::MatrixXd& points, KDTree& kd_tree,
    const Eigen::VectorXd &cp, const Eigen::MatrixXd& ma_centers,
    double k, double alpha
) {
    // Set of approximate medial points that correspond to the k nearest samples from p
    std::vector<size_t> indices(k);
    std::vector<double> dists(k);
    nanoflann::KNNResultSet<double> resultSet(k);
    resultSet.init(&indices[0], &dists[0]);
    kd_tree.index->findNeighbors(resultSet, &p[0], nanoflann::SearchParams(10)); 

    double max_radius = -1.0;
    for (size_t i = 0; i < k; ++i) {
        Eigen::VectorXd m = ma_centers.row(indices[i]).transpose();
        Eigen::VectorXd vpm = m - p;
        double cos_theta = cos_angle(vpm, cp-p);
        double esm_radius = alpha * vpm.norm() * cos_theta;
        max_radius = std::max(esm_radius, max_radius);
    }
    return max_radius;
}

void remove_invalid_points(
    Eigen::MatrixXd& ma_points,
    Eigen::VectorXd& ma_radii
) {
    size_t dim = ma_points.cols();
    std::vector<Eigen::VectorXd> points;
    std::vector<double> radii;
    // Store only valid points and radii (whose radius is not negative)
    for (size_t i = 0; i < ma_points.rows(); ++i) {
        if (ma_radii(i) < 0.0) continue;
        points.push_back(ma_points.row(i).transpose());
        radii.push_back(ma_radii(i));
    }
    // Convert back to Eigen Matrices
    ma_points.resize(points.size(), dim); ma_points.setZero();
    ma_radii.resize(radii.size()); ma_radii.setZero();
    #pragma omp parallel for
    for (size_t i = 0; i < points.size(); ++i) {
        ma_points.row(i) = points[i].transpose();
        ma_radii(i) = radii[i];
    }
}


void compute_medial_axis(
    Eigen::MatrixXd& ma_points,
    Eigen::VectorXd& ma_radii,
    const Eigen::MatrixXd& points,
    const Eigen::MatrixXd& normals,
    int max_iters=30,
    double eps=1e-4,
    double ndot_thd=0.99,
    double denoise_k=6,
    double denoise_alpha=0.7 
) {
    assert(points.rows() == normals.rows());
    assert(points.cols() == normals.cols());
    assert((points.cols() == 2) || (points.cols() == 3));

    Eigen::MatrixXd _normals(normals);
    // Make sure each normal vector is normalized
    for (int i = 0; i < _normals.rows(); ++i) {
        _normals.row(i).normalize();
    }

    // Approximate the maximum distance in point cloud
    Eigen::VectorXd minValues = points.colwise().minCoeff();
    Eigen::VectorXd maxValues = points.colwise().maxCoeff();
    double init_r = std::sqrt((maxValues - minValues).squaredNorm());

    // Build KD-tree for KNN
    KDTree kd_tree(points.cols(), points, 10);
    kd_tree.index->buildIndex();

    // Approximate the distance from one point to its neighbour
    //  assuming a uniform sampling
    std::vector<size_t> indices(2);
    std::vector<double> dists(2);
    nanoflann::KNNResultSet<double> resultSet(2);
    resultSet.init(&indices[0], &dists[0]);
    Eigen::VectorXd p0 = points.row(0).transpose();
    kd_tree.index->findNeighbors(resultSet, &p0[0], nanoflann::SearchParams(10));
    Eigen::VectorXd q0 = points.row(indices[1]).transpose();
    double neib_dist = (p0 - q0).norm();

    // Process each point
    ma_points.resize(points.rows(), points.cols()); ma_points.setZero();
    ma_radii.resize(points.rows()); ma_radii.setZero();
    #pragma omp parallel for
    for (size_t i = 0; i < points.rows(); ++i) {
        Eigen::VectorXd p = points.row(i).transpose();
        Eigen::VectorXd n = _normals.row(i).transpose();
        auto [c, r] = compute_single_ma_point(
            points, _normals,
            kd_tree, i,
            init_r, neib_dist,
            max_iters, eps, ndot_thd
        );
        ma_points.row(i) = c;
        ma_radii(i) = r;
    }
    // Denoising
    #pragma omp parallel for
    for (size_t i = 0; i < points.rows(); ++i) {
        double r = ma_radii(i);
        if (r < 0.0) continue;
        Eigen::VectorXd cp = ma_points.row(i).transpose();
        Eigen::VectorXd p = points.row(i).transpose();
        double rho = expected_radius(p, points, kd_tree, cp, ma_points, denoise_k, denoise_alpha);
        if (r < rho) {
            ma_radii(i) = -1.0;
        }
    }
    remove_invalid_points(ma_points, ma_radii);
}


#endif  // MASB_COMPUTE_MEDIAL_AXIS