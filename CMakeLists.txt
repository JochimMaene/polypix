cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME polypix)
endif()
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
    set(SKBUILD_PROJECT_VERSION 0.1.0)
endif()

project(
    "${SKBUILD_PROJECT_NAME}"
    VERSION "${SKBUILD_PROJECT_VERSION}"
    LANGUAGES CXX
)

if(WIN32)
    message(FATAL_ERROR
        "Polypix does not currently support Windows builds. "
        "Use Linux or macOS with healpix_cxx from the active build environment.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(POLYPIX_STABLE_ABI "Build an abi3 extension using Python's stable ABI. Requires Python 3.12+." OFF)
option(POLYPIX_BUNDLE_RUNTIME_DEPS "Bundle native runtime dependencies during CMake install. Release wheels should use auditwheel/delocate/delvewheel instead." OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT nanobind_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE POLYPIX_NANOBIND_DIR
        RESULT_VARIABLE POLYPIX_NANOBIND_DIR_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(POLYPIX_NANOBIND_DIR_RESULT EQUAL 0 AND EXISTS "${POLYPIX_NANOBIND_DIR}")
        set(nanobind_DIR "${POLYPIX_NANOBIND_DIR}" CACHE PATH "nanobind CMake directory")
    endif()
endif()

find_package(nanobind CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(HEALPIX_CXX REQUIRED IMPORTED_TARGET healpix_cxx)
pkg_check_modules(CFITSIO REQUIRED IMPORTED_TARGET cfitsio)

if(APPLE)
    foreach(property IN ITEMS INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_OPTIONS)
        get_target_property(healpix_options PkgConfig::HEALPIX_CXX "${property}")
        if(healpix_options)
            list(REMOVE_ITEM healpix_options "-fopenmp")
            set_target_properties(PkgConfig::HEALPIX_CXX PROPERTIES
                "${property}" "${healpix_options}"
            )
        endif()
    endforeach()
endif()

set(POLYPIX_SOURCES
    src/polypix/module.cpp
    src/polypix/core.cpp
)

set(POLYPIX_NANOBIND_FLAGS NB_STATIC)
if(POLYPIX_STABLE_ABI)
    if(Python_VERSION VERSION_LESS 3.12)
        message(FATAL_ERROR "POLYPIX_STABLE_ABI requires Python 3.12 or newer.")
    endif()
    list(APPEND POLYPIX_NANOBIND_FLAGS STABLE_ABI)
endif()

nanobind_add_module(_core ${POLYPIX_NANOBIND_FLAGS} ${POLYPIX_SOURCES})

target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(_core PRIVATE PkgConfig::HEALPIX_CXX PkgConfig::CFITSIO)

set_target_properties(_core PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_core"
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_WITH_INSTALL_RPATH OFF
    INSTALL_REMOVE_ENVIRONMENT_RPATH ON
)

if(APPLE)
    set(POLYPIX_EXTENSION_RPATH "@loader_path/../polypix.libs")
elseif(UNIX)
    set(POLYPIX_EXTENSION_RPATH "$ORIGIN/../polypix.libs")
else()
    set(POLYPIX_EXTENSION_RPATH "")
endif()

if(POLYPIX_BUNDLE_RUNTIME_DEPS AND POLYPIX_EXTENSION_RPATH)
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "${POLYPIX_EXTENSION_RPATH}"
    )
else()
    set_target_properties(_core PROPERTIES INSTALL_RPATH "")
endif()

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/polypix"
)

install(FILES
    polypix/_core.pyi
    polypix/py.typed
    polypix/__init__.pyi
    DESTINATION polypix
)

if(POLYPIX_BUNDLE_RUNTIME_DEPS)
    install(TARGETS _core
        RUNTIME_DEPENDENCY_SET polypix_runtime_dependencies
        LIBRARY DESTINATION polypix
        RUNTIME DESTINATION polypix
    )
    install(RUNTIME_DEPENDENCY_SET polypix_runtime_dependencies
        PRE_EXCLUDE_REGEXES
            "api-ms-"
            "ext-ms-"
        POST_EXCLUDE_REGEXES
            "^/lib/"
            "^/lib64/"
            "^/usr/lib/"
            "^/usr/lib64/"
            "^/System/Library/"
            "^/usr/lib/"
        LIBRARY DESTINATION polypix.libs
        RUNTIME DESTINATION polypix.libs
        FRAMEWORK DESTINATION polypix.libs
    )
else()
    install(TARGETS _core LIBRARY DESTINATION polypix RUNTIME DESTINATION polypix)
    if(UNIX)
        install(CODE [[
            file(GLOB POLYPIX_INSTALLED_EXTENSIONS
                "${CMAKE_INSTALL_PREFIX}/polypix/_core*.so"
                "${CMAKE_INSTALL_PREFIX}/polypix/_core*.dylib")
            foreach(POLYPIX_INSTALLED_EXTENSION IN LISTS POLYPIX_INSTALLED_EXTENSIONS)
                file(RPATH_REMOVE FILE "${POLYPIX_INSTALLED_EXTENSION}")
            endforeach()
        ]])
    endif()
endif()
install(FILES polypix/__init__.py polypix/bench.py DESTINATION polypix)
