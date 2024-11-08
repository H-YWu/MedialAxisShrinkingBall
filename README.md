# Shrinking Ball Algorithm for Approximating Medial Axis

C++, OpenMP implementation for 2D and 3D cases of the shrinking ball algorithm \[1\].

## Build

You should have _Eigen_, _cnpy_, and _nanoflann_ installed on your machine.

```bash
mkdir build && cd build
cmake ..
make
```

## Example

1. Generate _.npy_ files as input (point cloud & normals) for the shrinking ball algorithm:

```bash
./xyz2npy ../data/elephant
```

2. Generate _.xyz_ files (center of medial balls) using the shrinking ball algorithm:

```bash
./masb ../data/elephant
```

## References

\[1\] Ma, J., Bae, S.W. & Choi, S. 3D medial axis point approximation using nearest neighbors and the normal field. Vis Comput 28, 7–19 (2012). https://doi.org/10.1007/s00371-011-0594-7

\[2\] Ravi Peters, Kevin Wiebe, & Jeff Coukell. masbcpp. GitHub (2016). https://github.com/tudelft3d/masbcpp