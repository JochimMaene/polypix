#include "core.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

#define STRINGIFY(value) #value
#define MACRO_STRINGIFY(value) STRINGIFY(value)

namespace {

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> vector_to_array(std::vector<T> values) {
    auto* owner_vector = new std::vector<T>(std::move(values));

    nb::capsule owner(owner_vector, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });

    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(
        owner_vector->data(),
        {owner_vector->size()},
        owner);
}

nb::ndarray<nb::numpy, double, nb::ndim<2>> lonlat_to_array(
    std::vector<std::array<double, 2>> values) {
    auto* owner_vector = new std::vector<std::array<double, 2>>(std::move(values));

    nb::capsule owner(owner_vector, [](void* pointer) noexcept {
        delete static_cast<std::vector<std::array<double, 2>>*>(pointer);
    });

    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
        owner_vector->empty() ? nullptr : owner_vector->front().data(),
        {owner_vector->size(), static_cast<std::size_t>(2)},
        owner);
}

nb::ndarray<nb::numpy, double, nb::ndim<3>> boundaries_to_array(
    std::vector<std::array<double, 2>> values,
    std::size_t cell_count) {
    auto* owner_vector = new std::vector<std::array<double, 2>>(std::move(values));
    const auto corners_per_cell = cell_count == 0
        ? static_cast<std::size_t>(4 * kBoundaryStep)
        : owner_vector->size() / cell_count;

    nb::capsule owner(owner_vector, [](void* pointer) noexcept {
        delete static_cast<std::vector<std::array<double, 2>>*>(pointer);
    });

    return nb::ndarray<nb::numpy, double, nb::ndim<3>>(
        owner_vector->empty() ? nullptr : owner_vector->front().data(),
        {cell_count, corners_per_cell, static_cast<std::size_t>(2)},
        owner);
}

nb::dict coverage_to_python(CoverageResultNative coverage) {
    nb::dict result;
    result["offsets"] = vector_to_array(std::move(coverage.cell_offsets));
    result["cell_ids"] = vector_to_array(std::move(coverage.cell_ids));
    return result;
}

}  // namespace

NB_MODULE(_core, module) {
    module.doc() = "Polypix HEALPix center-in-polygon coverage kernel.";

#ifdef VERSION_INFO
    module.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    module.attr("__version__") = "dev";
#endif

    module.def(
        "_cover",
        [](const DoubleMatrix& vertices,
           const UInt64Vector& polygon_offsets,
           int resolution) {
            return coverage_to_python(cover_many_healpix(
                load_xyz_batch(vertices, polygon_offsets), resolution));
        },
        "vertices_xyz"_a.noconvert(), "offsets"_a.noconvert(), "resolution"_a);

    module.def(
        "_cover_lonlat",
        [](const DoubleMatrix& vertices,
           const UInt64Vector& polygon_offsets,
           int resolution) {
            return coverage_to_python(cover_many_healpix(
                load_lonlat_batch(vertices, polygon_offsets), resolution));
        },
        "vertices_lonlat_deg"_a.noconvert(), "offsets"_a.noconvert(), "resolution"_a);


    module.def(
        "_center",
        [](const UInt64Vector& cell_ids) {
            std::vector<std::array<double, 2>> lonlat;
            {
                nb::gil_scoped_release release;
                lonlat = cell_centers_lonlat(
                    cell_ids.data(), static_cast<std::size_t>(cell_ids.shape(0)));
            }
            return lonlat_to_array(std::move(lonlat));
        },
        "cell_ids"_a.noconvert());

    module.def(
        "_boundary",
        [](std::uint64_t cell_id) {
            std::vector<std::array<double, 2>> lonlat;
            {
                nb::gil_scoped_release release;
                lonlat = cell_boundary_lonlat(cell_id);
            }
            return lonlat_to_array(std::move(lonlat));
        },
        "cell_id"_a);

    module.def(
        "_boundary_many",
        [](const UInt64Vector& cell_ids) {
            const auto cell_count = static_cast<std::size_t>(cell_ids.shape(0));
            std::vector<std::array<double, 2>> lonlat;
            {
                nb::gil_scoped_release release;
                lonlat = cell_boundaries_lonlat(cell_ids.data(), cell_count);
            }
            return boundaries_to_array(std::move(lonlat), cell_count);
        },
        "cell_ids"_a.noconvert());
}
