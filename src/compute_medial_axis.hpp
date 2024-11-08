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
    int i, const Eigen::VectorXd& p, const Eigen::VectorXd& n,
    const Eigen::MatrixXd& points, KDTree& kd_tree,
    int max_iters, double eps
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, points.rows() - 1);
    int qidx;
    do { qidx = distrib(gen); } while (qidx == i);
    Eigen::VectorXd q = points.row(qidx).transpose();
    double r = compute_radius(p, n, q);

    unsigned int j = 0;
    Eigen::VectorXd c;
    while (j < max_iters) {
        c = p - r * n;

        // Nearest point q in cloud to current center
        std::vector<size_t> indices(2);
        std::vector<double> dists(2);
        nanoflann::KNNResultSet<double> resultSet(2);
        resultSet.init(&indices[0], &dists[0]);
        kd_tree.index->findNeighbors(resultSet, &c[0], nanoflann::SearchParams(10));

        qidx = indices[0];
        if (qidx== i) {
            q = points.row(indices[1]).transpose();
            Eigen::VectorXd vqp = p - q;
            if (cos_angle(vqp, n) > 0.0) qidx = indices[1];
            else return {Eigen::VectorXd::Zero(p.size()), -1.0};
        }

        Eigen::VectorXd q = points.row(qidx).transpose();
        double r_nxt = compute_radius(p, n, q);
        if (abs(r - r_nxt) < eps) break;

        r = r_nxt;
        ++j;
    }

    if (j > 0) return {c, r};
    return {Eigen::VectorXd::Zero(p.size()), -1.0};
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


void compute_medial_axis(
    Eigen::MatrixXd& ma_points,
    Eigen::VectorXd& ma_radii,
    const Eigen::MatrixXd& points,
    const Eigen::MatrixXd& normals,
    int max_iters=30,
    double eps=1e-8,
    double denoise_k=6,
    double denoise_alpha=0.7
) {
    assert(points.rows() == normals.rows());
    assert(points.cols() == normals.cols());
    assert((points.cols() == 2) || (points.cols() == 3));

    // Build KD-tree for KNN
    KDTree kd_tree(points.cols(), points, 10);
    kd_tree.index->buildIndex();

    // Process each point
    Eigen::MatrixXd ball_centers(points.rows(), points.cols());
    Eigen::VectorXd ball_radii(points.rows());
    #pragma omp parallel for
    for (size_t i = 0; i < points.rows(); ++i) {
        Eigen::VectorXd p = points.row(i).transpose();
        Eigen::VectorXd n = normals.row(i).transpose();
        auto result = compute_single_ma_point(i, p, n, points, kd_tree, max_iters, eps);
        ball_centers.row(i) = result.first;
        ball_radii(i) = result.second;
    }

    // Denoise and remove all inf points
    std::vector<Eigen::VectorXd> centers;
    std::vector<double> radii;
    #pragma omp parallel
    {
        std::vector<Eigen::Vector3d> local_centers;
        std::vector<double> local_radii;

        #pragma omp for nowait
        for (size_t i = 0; i < ball_centers.rows(); ++i) {
            double rp = ball_radii[i];
            if (rp > 0.0) {
                Eigen::VectorXd p = points.row(i).transpose();
                Eigen::VectorXd cp = ball_centers.row(i).transpose();
                double rho = expected_radius(p, points, kd_tree, cp, ball_centers, denoise_k, denoise_alpha);
                if (rp >= rho) {
                    local_centers.push_back(cp);
                    local_radii.push_back(rp);
                }
            }
        }

        #pragma omp critical
        {
            centers.insert(centers.end(), local_centers.begin(), local_centers.end());
            radii.insert(radii.end(), local_radii.begin(), local_radii.end());
        }
    }

    // Convert back to Eigen Matrices
    ma_points.resize(centers.size(), points.cols()); ma_points.setZero();
    ma_radii.resize(centers.size()); ma_radii.setZero();
    #pragma omp parallel for
    for (size_t i = 0; i < centers.size(); ++i) {
        ma_points.row(i) = centers[i];
        ma_radii(i) = radii[i];
    }
}

#endif  // MASB_COMPUTE_MEDIAL_AXIS