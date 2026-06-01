#pragma once

#include <healpix_cxx/pointing.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace nb = nanobind;

using DoubleMatrix =
    nb::ndarray<const double, nb::numpy, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using UInt64Vector =
    nb::ndarray<const std::uint64_t, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

struct Vec3 {
    double x;
    double y;
    double z;
};

struct PolygonBatch {
    std::vector<std::vector<pointing>> polygons;

    std::size_t polygon_count() const;
    const std::vector<pointing>& polygon(std::size_t index) const;
};

struct CoverageResultNative {
    int resolution = 0;
    std::size_t total_candidate_cells = 0;
    std::vector<std::uint64_t> cell_offsets;
    std::vector<std::uint64_t> cell_ids;
};

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegreesToRadians = kPi / 180.0;
constexpr double kRadiansToDegrees = 180.0 / kPi;
constexpr double kZeroNormEpsilon = 1e-15;
constexpr double kContainmentEpsilon = 1e-14;
constexpr double kUnitVectorTolerance = 1e-6;
constexpr int kMaxResolution = 29;
constexpr int kBoundaryStep = 1;

double dot(const Vec3& lhs, const Vec3& rhs);
Vec3 cross(const Vec3& lhs, const Vec3& rhs);
double norm(const Vec3& value);
Vec3 normalize(Vec3 value);
bool nearly_equal(const Vec3& lhs, const Vec3& rhs);
Vec3 from_lonlat_deg(double lon_deg, double lat_deg);
std::pair<double, double> to_lonlat_deg(const Vec3& vector);

std::vector<Vec3> normalize_convex_polygon(std::vector<Vec3> polygon);

PolygonBatch load_xyz_batch(
    const DoubleMatrix& vertices,
    const UInt64Vector& polygon_offsets);
PolygonBatch load_lonlat_batch(
    const DoubleMatrix& vertices,
    const UInt64Vector& polygon_offsets);

int resolution_to_nside(int resolution);
std::uint64_t encode_cell_id(int resolution, std::uint64_t nested_index);
std::pair<int, std::uint64_t> decode_cell_id(std::uint64_t cell_id);
std::pair<double, double> cell_center_lonlat(std::uint64_t cell_id);
std::vector<std::array<double, 2>> cell_centers_lonlat(
    const std::uint64_t* cell_ids,
    std::size_t cell_count);
std::vector<std::array<double, 2>> cell_boundary_lonlat(std::uint64_t cell_id);
std::vector<std::array<double, 2>> cell_boundaries_lonlat(
    const std::uint64_t* cell_ids,
    std::size_t cell_count);

CoverageResultNative cover_many_healpix(const PolygonBatch& batch, int resolution);
