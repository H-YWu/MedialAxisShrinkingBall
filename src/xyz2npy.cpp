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

#include "cnpy.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_xyz_dir>" << std::endl;
        return 1;
    }

    std::string dir = argv[1];
    std::string filename = dir + "/data.xyz";
    std::ifstream infile(filename);

    if (!infile) {
        std::cerr << "Failed to open XYZ file: " << filename << std::endl;
        return 1;
    }

    std::vector<double> coords;
    std::vector<double> normals;

    double x, y, z, nx, ny, nz;
    while (infile >> x >> y >> z >> nx >> ny >> nz) {
        coords.push_back(x);
        coords.push_back(y);
        coords.push_back(z);
        normals.push_back(nx);
        normals.push_back(ny);
        normals.push_back(nz);
    }
    infile.close();

    size_t num_points = coords.size() / 3;
    cnpy::npy_save(dir + "/coords.npy", coords.data(), {num_points, 3}, "w");
    cnpy::npy_save(dir + "/normals.npy", normals.data(), {num_points, 3}, "w");

    std::cout << "Files 'coords.npy' and 'normals.npy' have been saved." << std::endl;
    return 0;
}
