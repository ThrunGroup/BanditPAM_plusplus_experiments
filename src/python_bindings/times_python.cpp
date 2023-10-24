/**
 * @file cache_python.cpp
 * @date 2021-08-16
 *
 * Defines the functions to cache statistics in the Python bindings.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_pywrapper.hpp"

namespace km {

  size_t km::KMedoidsWrapper::getTotalBuildTimePython() {
    return KMedoids::getTotalBuildTime();
  }

  size_t km::KMedoidsWrapper::getTotalSwapTimePython() {
    return KMedoids::getTotalSwapTime();
  }

  size_t km::KMedoidsWrapper::getTimePerSwapPython() {
    return KMedoids::getTimePerSwap();
  }

  size_t km::KMedoidsWrapper::getTotalTimePython() {
    return KMedoids::getTotalTime();
  }

  void total_build_time(pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def_property_readonly(
            "total_build_time",
            &KMedoidsWrapper::getTotalBuildTimePython);
  }

  void total_swap_time_python(pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def_property_readonly(
            "total_swap_time",
            &KMedoidsWrapper::getTotalSwapTimePython);
  }

  void time_per_swap_python(pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def_property_readonly(
            "time_per_swap",
            &KMedoidsWrapper::getTimePerSwapPython);
  }

  void total_time(pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def_property_readonly(
            "total_time",
            &KMedoidsWrapper::getTotalTimePython);
  }
}  // namespace km
