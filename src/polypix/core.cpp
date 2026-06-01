#include "core.h"

#include <healpix_cxx/healpix_base.h>
#include <healpix_cxx/pointing.h>
#include <healpix_cxx/rangeset.h>
#include <healpix_cxx/vec3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace {

int highest_bit_index(std::uint64_t value) {
    if (value == 0) return -1;
    return 63 - __builtin_clzll(value);
}

void validate_resolution(int resolution) {
    if (resolution < 0 || resolution > kMaxResolution) {
        throw std::invalid_argument(
            "resolution must be between 0 and " + std::to_string(kMaxResolution) + ".");
    }
}

std::uint64_t pixel_count_for_resolution(int resolution) {
    validate_resolution(resolution);
    return 12ULL << (2 * resolution);
}

Vec3 normalize_input_vector(Vec3 value) {
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
        throw std::invalid_argument("Unit-vector inputs must be finite.");
    }
    const double length = norm(value);
    if (length <= kZeroNormEpsilon) {
        throw std::invalid_argument("Encountered a zero-length vector.");
    }
    if (std::abs(length - 1.0) > kUnitVectorTolerance) {
        throw std::invalid_argument(
            "Unit-vector inputs must already be normalized within tolerance.");
    }
    return normalize(value);
}

std::vector<std::uint64_t> load_offsets(
    const UInt64Vector& polygon_offsets,
    std::size_t vertex_count) {
    if (polygon_offsets.ndim() != 1) {
        throw std::invalid_argument("polygon_offsets must be a one-dimensional uint64 array.");
    }
    if (polygon_offsets.shape(0) == 0) {
        throw std::invalid_argument("polygon_offsets must contain at least one value.");
    }

    std::vector<std::uint64_t> offsets(static_cast<std::size_t>(polygon_offsets.shape(0)));
    std::memcpy(offsets.data(), polygon_offsets.data(), offsets.size() * sizeof(std::uint64_t));

    if (offsets.front() != 0) {
        throw std::invalid_argument("polygon_offsets must start at 0.");
    }
    if (offsets.back() != vertex_count) {
        throw std::invalid_argument("polygon_offsets must end at the total vertex count.");
    }
    for (std::size_t index = 1; index < offsets.size(); ++index) {
        if (offsets[index] < offsets[index - 1]) {
            throw std::invalid_argument("polygon_offsets must be nondecreasing.");
        }
    }
    return offsets;
}

pointing to_healpix_pointing(const Vec3& vector) {
    double phi = std::atan2(vector.y, vector.x);
    if (phi < 0.0) {
        phi += 2.0 * kPi;
    }
    return pointing(std::acos(std::clamp(vector.z, -1.0, 1.0)), phi);
}

Vec3 from_healpix_vec3(const vec3& vector) {
    return normalize(Vec3{vector.x, vector.y, vector.z});
}

constexpr std::size_t kMinParallelPolygons = 256;

unsigned int automatic_worker_limit(int resolution) {
    if (resolution <= 6) {
        return 4;
    }
    if (resolution == 7) {
        return 8;
    }
    return 16;
}

unsigned int configured_worker_count(int resolution) {
    const auto* value = std::getenv("POLYPIX_NUM_THREADS");
    if (value == nullptr || value[0] == '\0') {
        const auto hardware_workers = std::thread::hardware_concurrency();
        return hardware_workers == 0
            ? 1
            : std::min(hardware_workers, automatic_worker_limit(resolution));
    }

    char* end = nullptr;
    const auto parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 1;
    }
    return static_cast<unsigned int>(parsed);
}

struct CoverageChunk {
    std::vector<std::uint64_t> cell_ids;
    std::vector<std::uint64_t> cell_counts;
    std::size_t total_candidate_cells = 0;
};

void append_polygon_cells(
    const std::vector<pointing>& polygon,
    T_Healpix_Base<int64>& healpix,
    int resolution,
    CoverageChunk& chunk) {
    rangeset<int64> pixel_ranges;
    healpix.query_polygon(polygon, pixel_ranges);

    const auto polygon_cell_count = static_cast<std::size_t>(pixel_ranges.nval());
    chunk.total_candidate_cells += polygon_cell_count;
    chunk.cell_counts.push_back(static_cast<std::uint64_t>(polygon_cell_count));

    for (std::size_t range_index = 0; range_index < pixel_ranges.nranges(); ++range_index) {
        for (int64 pixel = pixel_ranges.ivbegin(range_index);
             pixel < pixel_ranges.ivend(range_index); ++pixel) {
            chunk.cell_ids.push_back(
                encode_cell_id(resolution, static_cast<std::uint64_t>(pixel)));
        }
    }
}

CoverageResultNative merge_chunks(
    std::vector<CoverageChunk>& chunks,
    int resolution,
    std::size_t polygon_count) {
    CoverageResultNative result;
    result.resolution = resolution;
    result.cell_offsets.reserve(polygon_count + 1);
    result.cell_offsets.push_back(0);

    std::size_t total_cell_count = 0;
    for (const auto& chunk : chunks) {
        total_cell_count += chunk.cell_ids.size();
        result.total_candidate_cells += chunk.total_candidate_cells;
    }
    result.cell_ids.reserve(total_cell_count);

    for (auto& chunk : chunks) {
        result.cell_ids.insert(
            result.cell_ids.end(),
            std::make_move_iterator(chunk.cell_ids.begin()),
            std::make_move_iterator(chunk.cell_ids.end()));
        for (const auto count : chunk.cell_counts) {
            result.cell_offsets.push_back(result.cell_offsets.back() + count);
        }
    }

    return result;
}

CoverageResultNative cover_many_healpix_sequential(
    const PolygonBatch& batch,
    int resolution,
    std::size_t start,
    std::size_t end) {
    std::vector<CoverageChunk> chunks(1);
    chunks.front().cell_counts.reserve(end - start);

    T_Healpix_Base<int64> healpix(resolution_to_nside(resolution), NEST, SET_NSIDE);
    for (std::size_t polygon_index = start; polygon_index < end; ++polygon_index) {
        append_polygon_cells(batch.polygon(polygon_index), healpix, resolution, chunks.front());
    }

    return merge_chunks(chunks, resolution, end - start);
}

}  // namespace

std::size_t PolygonBatch::polygon_count() const { return polygons.size(); }

const std::vector<pointing>& PolygonBatch::polygon(std::size_t index) const {
    if (index >= polygons.size()) {
        throw std::out_of_range("Polygon index out of range.");
    }
    return polygons[index];
}

double dot(const Vec3& lhs, const Vec3& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Vec3 cross(const Vec3& lhs, const Vec3& rhs) {
    return Vec3{
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

double norm(const Vec3& value) { return std::sqrt(dot(value, value)); }

Vec3 normalize(Vec3 value) {
    const double length = norm(value);
    if (length <= kZeroNormEpsilon) {
        throw std::invalid_argument("Encountered a zero-length vector.");
    }
    value.x /= length;
    value.y /= length;
    value.z /= length;
    return value;
}

bool nearly_equal(const Vec3& lhs, const Vec3& rhs) {
    return std::abs(lhs.x - rhs.x) < 1e-12 && std::abs(lhs.y - rhs.y) < 1e-12 &&
           std::abs(lhs.z - rhs.z) < 1e-12;
}

Vec3 from_lonlat_deg(double lon_deg, double lat_deg) {
    if (!std::isfinite(lon_deg) || !std::isfinite(lat_deg)) {
        throw std::invalid_argument("lon/lat inputs must be finite.");
    }
    if (lat_deg < -90.0 || lat_deg > 90.0) {
        throw std::invalid_argument("latitude must be between -90 and 90 degrees.");
    }
    const double lon = lon_deg * kDegreesToRadians;
    const double lat = lat_deg * kDegreesToRadians;
    const double cos_lat = std::cos(lat);
    return Vec3{
        cos_lat * std::cos(lon),
        cos_lat * std::sin(lon),
        std::sin(lat),
    };
}

std::pair<double, double> to_lonlat_deg(const Vec3& vector) {
    const auto normalized = normalize(vector);
    const double lon = std::atan2(normalized.y, normalized.x) * kRadiansToDegrees;
    const double lat =
        std::atan2(normalized.z, std::hypot(normalized.x, normalized.y)) * kRadiansToDegrees;
    return {lon, lat};
}

std::vector<Vec3> normalize_convex_polygon(std::vector<Vec3> polygon) {
    if (polygon.size() < 3) {
        throw std::invalid_argument("Each polygon needs at least three vertices.");
    }
    if (nearly_equal(polygon.front(), polygon.back())) {
        polygon.pop_back();
    }
    if (polygon.size() < 3) {
        throw std::invalid_argument("Each polygon needs at least three unique vertices.");
    }

    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const std::size_t next = (i + 1) % polygon.size();
        if (nearly_equal(polygon[i], polygon[next])) {
            throw std::invalid_argument("Polygon contains duplicate consecutive vertices.");
        }
        for (std::size_t j = i + 1; j < polygon.size(); ++j) {
            if (nearly_equal(polygon[i], polygon[j])) {
                throw std::invalid_argument("Polygon contains duplicate vertices.");
            }
        }
    }

    Vec3 interior{0.0, 0.0, 0.0};
    for (const auto& vertex : polygon) {
        interior.x += vertex.x;
        interior.y += vertex.y;
        interior.z += vertex.z;
    }
    if (norm(interior) <= kZeroNormEpsilon) {
        bool found_normal = false;
        for (std::size_t index = 0; index < polygon.size(); ++index) {
            const auto candidate = cross(polygon[index], polygon[(index + 1) % polygon.size()]);
            if (norm(candidate) > kZeroNormEpsilon) {
                interior = candidate;
                found_normal = true;
                break;
            }
        }
        if (!found_normal) {
            throw std::invalid_argument("Polygon is degenerate.");
        }
    }
    interior = normalize(interior);

    double orientation = 0.0;
    for (std::size_t index = 0; index < polygon.size(); ++index) {
        const auto& current = polygon[index];
        const auto& next = polygon[(index + 1) % polygon.size()];
        orientation += dot(cross(current, next), interior);
    }
    if (std::abs(orientation) <= kContainmentEpsilon) {
        throw std::invalid_argument("Polygon is degenerate or numerically ambiguous.");
    }
    if (orientation < 0.0) {
        std::reverse(polygon.begin(), polygon.end());
    }

    for (std::size_t edge_index = 0; edge_index < polygon.size(); ++edge_index) {
        const auto& current = polygon[edge_index];
        const auto& next = polygon[(edge_index + 1) % polygon.size()];
        const auto edge_normal = cross(current, next);
        if (norm(edge_normal) <= kZeroNormEpsilon) {
            throw std::invalid_argument("Polygon contains degenerate edges.");
        }
        bool found_strict_interior = false;
        for (const auto& vertex : polygon) {
            const double side = dot(edge_normal, vertex);
            if (side < -kContainmentEpsilon) {
                throw std::invalid_argument("Polygon must be convex.");
            }
            if (side > kContainmentEpsilon) {
                found_strict_interior = true;
            }
        }
        if (!found_strict_interior) {
            throw std::invalid_argument("Polygon is degenerate.");
        }
    }

    return polygon;
}

PolygonBatch load_xyz_batch(
    const DoubleMatrix& vertices,
    const UInt64Vector& polygon_offsets) {
    if (vertices.ndim() != 2 || vertices.shape(1) != 3) {
        throw std::invalid_argument("vertices_xyz must have shape (vertices, 3).");
    }

    const auto vertex_count = static_cast<std::size_t>(vertices.shape(0));
    const auto offsets = load_offsets(polygon_offsets, vertex_count);
    const auto* raw = vertices.data();
    PolygonBatch batch;
    batch.polygons.reserve(offsets.size() - 1);

    for (std::size_t polygon_index = 0; polygon_index + 1 < offsets.size(); ++polygon_index) {
        const auto start = static_cast<std::size_t>(offsets[polygon_index]);
        const auto end = static_cast<std::size_t>(offsets[polygon_index + 1]);
        if (end - start < 3) {
            throw std::invalid_argument("Each polygon needs at least three vertices.");
        }

        std::vector<Vec3> polygon;
        polygon.reserve(end - start);
        for (std::size_t vertex_index = start; vertex_index < end; ++vertex_index) {
            const auto offset = vertex_index * 3;
            polygon.push_back(normalize_input_vector(
                Vec3{raw[offset], raw[offset + 1], raw[offset + 2]}));
        }
        polygon = normalize_convex_polygon(std::move(polygon));

        std::vector<pointing> healpix_polygon;
        healpix_polygon.reserve(polygon.size());
        for (const auto& vertex : polygon) {
            healpix_polygon.push_back(to_healpix_pointing(vertex));
        }
        batch.polygons.push_back(std::move(healpix_polygon));
    }

    return batch;
}

PolygonBatch load_lonlat_batch(
    const DoubleMatrix& vertices,
    const UInt64Vector& polygon_offsets) {
    if (vertices.ndim() != 2 || vertices.shape(1) != 2) {
        throw std::invalid_argument("vertices_lonlat_deg must have shape (vertices, 2).");
    }

    const auto vertex_count = static_cast<std::size_t>(vertices.shape(0));
    const auto offsets = load_offsets(polygon_offsets, vertex_count);
    const auto* raw = vertices.data();
    PolygonBatch batch;
    batch.polygons.reserve(offsets.size() - 1);

    for (std::size_t polygon_index = 0; polygon_index + 1 < offsets.size(); ++polygon_index) {
        const auto start = static_cast<std::size_t>(offsets[polygon_index]);
        const auto end = static_cast<std::size_t>(offsets[polygon_index + 1]);
        if (end - start < 3) {
            throw std::invalid_argument("Each polygon needs at least three vertices.");
        }

        std::vector<Vec3> polygon;
        polygon.reserve(end - start);
        for (std::size_t vertex_index = start; vertex_index < end; ++vertex_index) {
            const auto offset = vertex_index * 2;
            polygon.push_back(from_lonlat_deg(raw[offset], raw[offset + 1]));
        }
        polygon = normalize_convex_polygon(std::move(polygon));

        std::vector<pointing> healpix_polygon;
        healpix_polygon.reserve(polygon.size());
        for (const auto& vertex : polygon) {
            healpix_polygon.push_back(to_healpix_pointing(vertex));
        }
        batch.polygons.push_back(std::move(healpix_polygon));
    }

    return batch;
}

int resolution_to_nside(int resolution) {
    validate_resolution(resolution);
    return 1 << resolution;
}

std::uint64_t encode_cell_id(int resolution, std::uint64_t nested_index) {
    const auto pixel_count = pixel_count_for_resolution(resolution);
    if (nested_index >= pixel_count) {
        throw std::invalid_argument("nested_index is out of range for the given resolution.");
    }
    const int payload_bits = 4 + 2 * resolution;
    return (1ULL << payload_bits) | nested_index;
}

std::pair<int, std::uint64_t> decode_cell_id(std::uint64_t cell_id) {
    if (cell_id < (1ULL << 4)) {
        throw std::invalid_argument("cell_id is not a valid packed HEALPix token.");
    }
    const int payload_bits = highest_bit_index(cell_id);
    if (payload_bits < 4 || ((payload_bits - 4) % 2) != 0) {
        throw std::invalid_argument("cell_id is not a valid packed HEALPix token.");
    }
    const int resolution = (payload_bits - 4) / 2;
    validate_resolution(resolution);
    const auto nested_index = cell_id ^ (1ULL << payload_bits);
    if (nested_index >= pixel_count_for_resolution(resolution)) {
        throw std::invalid_argument("cell_id is not a valid packed HEALPix token.");
    }
    return {resolution, nested_index};
}

std::pair<double, double> cell_center_lonlat(std::uint64_t cell_id) {
    const auto [resolution, nested_index] = decode_cell_id(cell_id);
    T_Healpix_Base<int64> healpix(resolution_to_nside(resolution), NEST, SET_NSIDE);
    return to_lonlat_deg(from_healpix_vec3(healpix.pix2vec(static_cast<int64>(nested_index))));
}

std::vector<std::array<double, 2>> cell_centers_lonlat(
    const std::uint64_t* cell_ids,
    std::size_t cell_count) {
    std::vector<std::array<double, 2>> lonlat;
    lonlat.reserve(cell_count);
    std::unordered_map<int, T_Healpix_Base<int64>> healpix_by_resolution;

    for (std::size_t index = 0; index < cell_count; ++index) {
        const auto [resolution, nested_index] = decode_cell_id(cell_ids[index]);
        auto [iterator, inserted] = healpix_by_resolution.try_emplace(
            resolution, resolution_to_nside(resolution), NEST, SET_NSIDE);
        (void)inserted;
        const auto [lon, lat] = to_lonlat_deg(
            from_healpix_vec3(iterator->second.pix2vec(static_cast<int64>(nested_index))));
        lonlat.push_back({lon, lat});
    }

    return lonlat;
}

std::vector<std::array<double, 2>> cell_boundary_lonlat(std::uint64_t cell_id) {
    const auto [resolution, nested_index] = decode_cell_id(cell_id);
    T_Healpix_Base<int64> healpix(resolution_to_nside(resolution), NEST, SET_NSIDE);
    std::vector<vec3> boundary;
    healpix.boundaries(static_cast<int64>(nested_index), kBoundaryStep, boundary);
    std::vector<std::array<double, 2>> lonlat;
    lonlat.reserve(boundary.size());
    for (const auto& corner : boundary) {
        const auto [lon, lat] = to_lonlat_deg(from_healpix_vec3(corner));
        lonlat.push_back({lon, lat});
    }
    return lonlat;
}

std::vector<std::array<double, 2>> cell_boundaries_lonlat(
    const std::uint64_t* cell_ids,
    std::size_t cell_count) {
    std::vector<std::array<double, 2>> lonlat;
    lonlat.reserve(cell_count * 4 * kBoundaryStep);
    std::unordered_map<int, T_Healpix_Base<int64>> healpix_by_resolution;
    std::vector<vec3> boundary;
    boundary.reserve(4 * kBoundaryStep);

    for (std::size_t index = 0; index < cell_count; ++index) {
        const auto [resolution, nested_index] = decode_cell_id(cell_ids[index]);
        auto [iterator, inserted] = healpix_by_resolution.try_emplace(
            resolution, resolution_to_nside(resolution), NEST, SET_NSIDE);
        (void)inserted;

        boundary.clear();
        iterator->second.boundaries(static_cast<int64>(nested_index), kBoundaryStep, boundary);
        for (const auto& corner : boundary) {
            const auto [lon, lat] = to_lonlat_deg(from_healpix_vec3(corner));
            lonlat.push_back({lon, lat});
        }
    }

    return lonlat;
}

CoverageResultNative cover_many_healpix(const PolygonBatch& batch, int resolution) {
    validate_resolution(resolution);

    const auto polygon_count = batch.polygon_count();
    const auto configured_workers = configured_worker_count(resolution);
    const auto worker_count = std::min<std::size_t>(configured_workers, polygon_count);

    nb::gil_scoped_release release;
    if (worker_count <= 1 || polygon_count < kMinParallelPolygons) {
        return cover_many_healpix_sequential(batch, resolution, 0, polygon_count);
    }

    std::vector<CoverageChunk> chunks(worker_count);
    std::vector<std::exception_ptr> exceptions(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
        const auto start = polygon_count * worker_index / worker_count;
        const auto end = polygon_count * (worker_index + 1) / worker_count;
        chunks[worker_index].cell_counts.reserve(end - start);
        workers.emplace_back([&, worker_index, start, end]() {
            try {
                T_Healpix_Base<int64> healpix(resolution_to_nside(resolution), NEST, SET_NSIDE);
                for (std::size_t polygon_index = start; polygon_index < end; ++polygon_index) {
                    append_polygon_cells(
                        batch.polygon(polygon_index),
                        healpix,
                        resolution,
                        chunks[worker_index]);
                }
            } catch (...) {
                exceptions[worker_index] = std::current_exception();
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    for (const auto& exception : exceptions) {
        if (exception) {
            std::rethrow_exception(exception);
        }
    }

    return merge_chunks(chunks, resolution, polygon_count);
}
