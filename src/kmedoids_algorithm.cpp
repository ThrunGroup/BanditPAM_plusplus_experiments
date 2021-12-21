/**
 * @file kmedoids_ucb.cpp
 * @date 2020-06-10
 *
 * This file contains the primary C++ implementation of the BanditPAM code.
 *
 */

#include "kmedoids_algorithm.hpp"
#include "fastpam1.hpp"
#include "pam.hpp"
#include "banditpam.hpp"

#include <armadillo>
#include <unordered_map>
#include <regex>
#include <omp.h>


/**
 *  \brief Class implementation for running KMedoids methods.
 *
 *  KMedoids class. Creates a KMedoids object that can be used to find the medoids
 *  for a particular set of input data.
 *
 *  @param n_medoids Number of medoids/clusters to create
 *  @param algorithm Algorithm used to find medoids; options are "BanditPAM" for
 *  the "Bandit-PAM" algorithm, or "naive" to use the naive method
 *  @param max_iter The maximum number of iterations the algorithm runs for
 *  @param buildConfidence Constant that affects the sensitivity of build confidence bounds
 *  @param swapConfidence Constant that affects the sensitiviy of swap confidence bounds
 */
km::KMedoids::KMedoids(size_t n_medoids, const std::string& algorithm,
                   size_t max_iter, size_t buildConfidence, size_t swapConfidence
    ): n_medoids(n_medoids),
       algorithm(algorithm),
       max_iter(max_iter),
       buildConfidence(buildConfidence),
       swapConfidence(swapConfidence) {
  km::KMedoids::checkAlgorithm(algorithm);
}

/**
 *  \brief Destroys KMedoids object.
 *
 *  Destructor for the KMedoids class.
 */
km::KMedoids::~KMedoids() {;} // TODO: Need semicolons?

double km::KMedoids::cachedLoss(const arma::mat& data, size_t i, size_t j, bool use_cache) {
  if (!use_cache) {
    return (this->*lossFn)(data, i, j);
  }
  
  size_t n = data.n_cols;
  size_t m = fmin(n, ceil(log10(data.n_cols) * cache_multiplier));
  
  if (reindex.find(j) != reindex.end()) { // test this is one of the early points in the permutation
    // TODO: Potential race condition with shearing?
    // T1 begins to write to cache and then T2 access in the middle of write?
    if (cache[(m*i) + reindex[j]] == -1) {
        cache[(m*i) + reindex[j]] = (this->*lossFn)(data, i, j);
    } 
    return cache[m*i + reindex[j]];
  }
  return (this->*lossFn)(data, i, j);
}

/**
 *  \brief Checks whether algorithm input is valid
 *
 *  Checks whether the user's selected algorithm is a valid option.
 *
 *  @param algorithm Name of the algorithm input by the user.
 */
void km::KMedoids::checkAlgorithm(const std::string& algorithm) {
  if ((algorithm != "BanditPAM") && (algorithm != "naive") && (algorithm != "FastPAM1")) {
    throw "unrecognized algorithm";
  }
}

/**
 *  \brief Returns the build medoids
 *
 *  Returns the build medoids at the end of the BUILD step after km::KMedoids::fit
 *  has been called.
 */
arma::urowvec km::KMedoids::getMedoidsBuild() {
  return medoid_indices_build;
}

/**
 *  \brief Returns the final medoids
 *
 *  Returns the final medoids at the end of the SWAP step after km::KMedoids::fit
 *  has been called.
 */
arma::urowvec km::KMedoids::getMedoidsFinal() {
  return medoid_indices_final;
}

/**
 *  \brief Returns the medoid assignments for each datapoint
 *
 *  Returns the medoid each input datapoint is assigned to after km::KMedoids::fit
 *  has been called and the final medoids have been identified
 */
arma::urowvec km::KMedoids::getLabels() {
  return labels;
}

/**
 *  \brief Returns the number of swap steps
 *
 *  Returns the number of SWAP steps completed during the last call to
 *  km::KMedoids::fit
 */
size_t km::KMedoids::getSteps() {
  return steps;
}

/**
 *  \brief Sets the loss function
 *
 *  Sets the loss function used during km::KMedoids::fit
 *
 *  @param loss Loss function to be used e.g. L2
 */
void km::KMedoids::setLossFn(std::string loss) {
  if (std::regex_match(loss, std::regex("L\\d*"))) {
      loss = loss.substr(1);
  }
  try {
    if (loss == "manhattan") {
        lossFn = &km::KMedoids::manhattan;
    } else if (loss == "cos") {
        lossFn = &km::KMedoids::cos;
    } else if (loss == "inf") {
        lossFn = &km::KMedoids::LINF;
    } else if (std::isdigit(loss.at(0))) {
        lossFn = &km::KMedoids::LP;
        lp     = atoi(loss.c_str());
    } else {
        throw std::invalid_argument("error: unrecognized loss function");
    }
  } catch (std::invalid_argument& e) {
      std::cout << e.what() << std::endl;
    }
}

/**
 *  \brief Returns the number of medoids
 *
 *  Returns the number of medoids to be identified during km::KMedoids::fit
 */
size_t km::KMedoids::getNMedoids() {
  return n_medoids;
}

/**
 *  \brief Sets the number of medoids
 *
 *  Sets the number of medoids to be identified during km::KMedoids::fit
 */
void km::KMedoids::setNMedoids(size_t new_num) {
  n_medoids = new_num;
}

/**
 *  \brief Returns the algorithm for KMedoids
 *
 *  Returns the algorithm used for identifying the medoids during km::KMedoids::fit
 */
std::string km::KMedoids::getAlgorithm() {
  return algorithm;
}

/**
 *  \brief Sets the algorithm for KMedoids
 *
 *  Sets the algorithm used for identifying the medoids during km::KMedoids::fit
 *
 *  @param new_alg New algorithm to use
 */
void km::KMedoids::setAlgorithm(const std::string& new_alg) {
  algorithm = new_alg;
  km::KMedoids::checkAlgorithm(algorithm);
}

/**
 *  \brief Returns the maximum number of iterations for KMedoids
 *
 *  Returns the maximum number of iterations that can be run during
 *  km::KMedoids::fit
 */
size_t km::KMedoids::getMaxIter() {
  return max_iter;
}

/**
 *  \brief Sets the maximum number of iterations for KMedoids
 *
 *  Sets the maximum number of iterations that can be run during km::KMedoids::fit
 *
 *  @param new_max New maximum number of iterations to use
 */
void km::KMedoids::setMaxIter(size_t new_max) {
  max_iter = new_max;
}

/**
 *  \brief Returns the constant buildConfidence
 *
 *  Returns the constant that affects the sensitivity of build confidence bounds
 *  that can be run during km::KMedoids::fit
 */
size_t km::KMedoids::getbuildConfidence() {
  return buildConfidence;
}

/**
 *  \brief Sets the constant buildConfidence
 *
 *  Sets the constant that affects the sensitivity of build confidence bounds
 *  that can be run during km::KMedoids::fit
 *
 *  @param new_buildConfidence New buildConfidence
 */
void km::KMedoids::setbuildConfidence(size_t new_buildConfidence) {
  buildConfidence = new_buildConfidence;
}

/**
 *  \brief Returns the constant swapConfidence
 *
 *  Returns the constant that affects the sensitivity of swap confidence bounds
 *  that can be run during km::KMedoids::fit
 */
size_t km::KMedoids::getswapConfidence() {
  return swapConfidence;
}

/**
 *  \brief Sets the constant swapConfidence
 *
 *  Sets the constant that affects the sensitivity of swap confidence bounds
 *  that can be run during km::KMedoids::fit
 *
 *  @param new_swapConfidence New swapConfidence
 */
void km::KMedoids::setswapConfidence(size_t new_swapConfidence) {
  swapConfidence = new_swapConfidence;
}

/**
 * \brief Finds medoids for the input data under identified loss function
 *
 * Primary function of the KMedoids class. Identifies medoids for input dataset
 * after both the SWAP and BUILD steps
 *
 * @param input_data Input data to find the medoids of
 * @param loss The loss function used during medoid computation
 */
void km::KMedoids::fit(const arma::mat& input_data, const std::string& loss) {
  batchSize = fmin(input_data.n_rows, batchSize);

  if (input_data.n_rows == 0) {
    throw std::invalid_argument("Dataset is empty");
  }

  km::KMedoids::setLossFn(loss);
  if (algorithm == "naive") {
    static_cast<PAM*>(this)->fit_naive(input_data);
  } else if (algorithm == "BanditPAM") {
    static_cast<BanditPAM*>(this)->fit_bpam(input_data);
  } else if (algorithm == "FastPAM1") {
    static_cast<FastPAM1*>(this)->fit_fastpam1(input_data);
  }
}

/**
 * \brief Calculates distances in swap step
 *
 * Calculates the best and second best distances for each datapoint to one of
 * the medoids in the current medoid set.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Array of medoid indices corresponding to dataset entries
 * @param best_distances Array of best distances from each point to previous set
 * of medoids
 * @param second_best_distances Array of second smallest distances from each
 * point to previous set of medoids
 * @param assignments Assignments of datapoints to their closest medoid
 */
void km::KMedoids::calc_best_distances_swap(
  const arma::mat& data,
  arma::urowvec& medoid_indices,
  arma::rowvec& best_distances,
  arma::rowvec& second_distances,
  arma::urowvec& assignments) {
#pragma omp parallel for
    for (size_t i = 0; i < data.n_cols; i++) {
        double best = std::numeric_limits<double>::infinity();
        double second = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < medoid_indices.n_cols; k++) {
            double cost = km::KMedoids::cachedLoss(data, i, medoid_indices(k));
            if (cost < best) {
                assignments(i) = k;
                second = best;
                best = cost;
            } else if (cost < second) {
                second = cost;
            }
        }
        best_distances(i) = best;
        second_distances(i) = second;
    }
}



/**
 * \brief Calculate loss for medoids
 *
 * Calculates the loss under the previously identified loss function of the
 * medoid indices.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Indices of the medoids in the dataset.
 */
double km::KMedoids::calc_loss(
  const arma::mat& data,
  arma::urowvec& medoid_indices) {
    double total = 0;

    // TODO: is this parallel loop accumulating properly?
    #pragma omp parallel for
    for (size_t i = 0; i < data.n_cols; i++) {
        double cost = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < n_medoids; k++) {
            double currCost = km::KMedoids::cachedLoss(data, i, medoid_indices(k));
            if (currCost < cost) {
                cost = currCost;
            }
        }
        total += cost;
    }
    return total;
}

// Loss and miscellaneous functions

/**
 * \brief LP loss
 *
 * Calculates the LP loss between the datapoints at index i and j of the dataset
 *
 * @param data Transposed input data to find the medoids of
 * @param i Index of first datapoint
 * @param j Index of second datapoint
 */
double km::KMedoids::LP(const arma::mat& data, size_t i, size_t j) const {
    return arma::norm(data.col(i) - data.col(j), lp);
}


/**
 * \brief cos loss
 *
 * Calculates the cosine loss between the datapoints at index i and j of the
 * dataset
 *
 * @param data Transposed input data to find the medoids of
 * @param i Index of first datapoint
 * @param j Index of second datapoint
 */
double km::KMedoids::cos(const arma::mat& data, size_t i, size_t j) const {
    return arma::dot(data.col(i), data.col(j)) / (arma::norm(data.col(i))
                                                    * arma::norm(data.col(j)));
}

/**
 * \brief Manhattan loss
 *
 * Calculates the Manhattan loss between the datapoints at index i and j of the
 * dataset
 *
 * @param data Transposed input data to find the medoids of
 * @param i Index of first datapoint
 * @param j Index of second datapoint
 */
double km::KMedoids::manhattan(const arma::mat& data, size_t i, size_t j) const {
    return arma::accu(arma::abs(data.col(i) - data.col(j)));
}

/**
 * \brief L_INFINITY loss
 *
 * Calculates the Manhattan loss between the datapoints at index i and j of the
 * dataset
 *
 * @param data Transposed input data to find the medoids of
 * @param i Index of first datapoint
 * @param j Index of second datapoint
 */
double km::KMedoids::LINF(const arma::mat& data, size_t i, size_t j) const {
    return arma::max(arma::abs(data.col(i) - data.col(j)));
}
