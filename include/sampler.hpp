#ifndef SAMPLER
#define SAMPLER

#include <random>
#include <cmath>
#include "neural_state.hpp"
#include "spin_system.hpp"

struct SamplerResult {
    int num_samples;
    int num_rejected;
    std::complex<float> avg_local_energy;
    Eigen::MatrixXcf weight_gradient;
    Eigen::VectorXcf visible_gradient;
    Eigen::VectorXcf hidden_gradient;
};

/* Sample gradients and local energy for a given nqstate and spin system. */

template <typename SpinSysT> /* Sublcass of spin system (e.g. Ising)*/
struct SamplerResult sample_spins(NeuralState nqstate, SpinSysT spin_sys, int nsweeps) {
    std::complex<float> old_psi, new_psi;
    float r, p;
    int num_samples, num_rejected;
    Eigen::VectorXcf spins;
    nqs_gradient grad;
    std::complex<float> s_psi;
    std::complex<float> avg_local_energy = std::complex<float>(0.0, 0.0);
    std::complex<float> local_energy = std::complex<float>(0.0, 0.0);
    // Averages of the network gradients normalized by the wave function value. 
    Eigen::MatrixXcf avg_weight_gradient = Eigen::MatrixXcf::Zero(nqstate.weights.rows(), nqstate.weights.cols());
    Eigen::VectorXcf avg_visible_gradient = Eigen::VectorXcf::Zero(nqstate.visible_bias.size());
    Eigen::VectorXcf avg_hidden_gradient = Eigen::VectorXcf::Zero(nqstate.hidden_bias.size());
    // Average of the local energy times the parameter gradient, normalized by the wave func value.
    Eigen::MatrixXcf avg_le_wgrad = Eigen::MatrixXcf::Zero(nqstate.weights.rows(), nqstate.weights.cols());
    Eigen::VectorXcf avg_le_visgrad = Eigen::VectorXcf::Zero(nqstate.visible_bias.size());
    Eigen::VectorXcf avg_le_hidgrad = Eigen::VectorXcf::Zero(nqstate.hidden_bias.size());
    // The total stochastic estimate of the gradient, to be used in the descent.
    Eigen::MatrixXcf weight_grad = Eigen::MatrixXcf::Zero(nqstate.weights.rows(), nqstate.weights.cols());
    Eigen::VectorXcf visible_grad = Eigen::VectorXcf::Zero(nqstate.visible_bias.size());
    Eigen::VectorXcf hidden_grad = Eigen::VectorXcf::Zero(nqstate.hidden_bias.size());

    spins = Eigen::VectorXcf::Constant(nqstate.num_visible, 1.0);

    num_samples = 0;
    num_rejected = 0;
    for (int k = 0; k < nsweeps; k++) {
        for (int i = 0; i < nqstate.num_visible; i++) {
            /* Metropolis-Hastings step. */
            old_psi = nqstate.evalute_state(spins);
            // Flip the i^th spin.
            spins(i) = std::complex<float>(-1.0, 0.0) * spins(i);
            new_psi = nqstate.evalute_state(spins);
            p = pow(std::abs(new_psi / old_psi), 2);
            r = (float)rand() / (float)RAND_MAX;
            if (r >= p) {
                // Flip rejected.
                spins(i) = std::complex<float>(-1.0, 0.0) * spins(i);
                num_rejected++;
            }
            num_samples++;

            /* Update all averages. */
            grad = nqstate.gradient(spins);//std::cout << spin_sys.local_energy(nqstate, spins) << std::endl;
            local_energy = spin_sys.local_energy(nqstate, spins);
            avg_local_energy += local_energy;
            avg_weight_gradient += grad.weight_grad;
            avg_visible_gradient += grad.visible_grad;
            avg_hidden_gradient += grad.hidden_grad;
            avg_le_wgrad += local_energy * grad.weight_grad;
            avg_le_visgrad += local_energy * grad.visible_grad;
            avg_le_hidgrad += local_energy * grad.hidden_grad;
        }
    }

    /* Divide averages by the number of samples */
    avg_local_energy = avg_local_energy / (float)num_samples;
    avg_weight_gradient = avg_weight_gradient / (float)num_samples;
    avg_visible_gradient = avg_visible_gradient / (float)num_samples;
    avg_hidden_gradient = avg_hidden_gradient / (float)num_samples;
    avg_le_wgrad = avg_le_wgrad / (float)num_samples;
    avg_le_visgrad = avg_le_visgrad / (float)num_samples;
    avg_le_hidgrad = avg_le_hidgrad / (float)num_samples;

    /* Compute the gradient form the collected samples. */
    weight_grad = 2.0 * (avg_le_wgrad - avg_local_energy * avg_weight_gradient);
    visible_grad = 2.0 * (avg_le_visgrad - avg_local_energy * avg_visible_gradient);
    hidden_grad = 2.0 * (avg_le_hidgrad - avg_local_energy * avg_hidden_gradient);

    SamplerResult result;
    result.num_samples = num_samples;
    result.num_rejected = num_rejected;
    result.avg_local_energy = avg_local_energy;
    result.weight_gradient = weight_grad;
    result.visible_gradient = visible_grad;
    result.hidden_gradient = hidden_grad;
    return result;
}

#endif