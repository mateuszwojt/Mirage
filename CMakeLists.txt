CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

cmake_policy(SET CMP0091 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# libMirage.a (and slang-rhi's static libs, fetched under mirage/kernels) end up
# linked into hdMirage.so, a shared object. macOS's toolchain emits
# position-independent code unconditionally, but on Linux static libs default to
# non-PIC, which fails the shared-object link with R_X86_64_TPOFF32 relocations
# against TLS variables. Building everything as PIC from the start avoids that.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LIB_MAJOR_VERSION "1")
set(LIB_MINOR_VERSION "2")
set(LIB_PATCH_VERSION "0")
set(LIB_VERSION_STRING "${LIB_MAJOR_VERSION}.${LIB_MINOR_VERSION}.${LIB_PATCH_VERSION}")

if (CMAKE_VERSION VERSION_LESS 3.0)
    PROJECT(Mirage CXX)
else()
    cmake_policy(SET CMP0048 NEW)
    PROJECT(Mirage VERSION "${LIB_VERSION_STRING}" LANGUAGES C CXX)
endif()

SET(CMAKE_MACOSX_RPATH FALSE)

SET(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include" CACHE PATH "The directory the headers are installed in")
SET(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib" CACHE STRING "Directory where lib will install")

IF(UNIX OR CYGWIN)
    SET(_CMAKE_INSTALL_DIR "${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}")
ELSEIF(WIN32)
    SET(_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/cmake")
ENDIF()
SET(CMAKE_INSTALL_DIR "${_CMAKE_INSTALL_DIR}" CACHE PATH "The directory cmake files are installed in")

option(MIRAGE_BUILD_TESTS "Build Mirage's unit/regression test suites (tests/) and register kernel_validate with CTest" ON)

# Must precede add_subdirectory(tools) so tools/kernel_validate's add_test()
# call is actually picked up by `ctest` - CMake only generates CTest
# machinery for directories processed after enable_testing() runs.
enable_testing()

add_subdirectory(mirage)
add_subdirectory(tools)

if(MIRAGE_BUILD_TESTS)
    add_subdirectory(tests)
endif()

EXPORT( PACKAGE ${PROJECT_NAME} )
