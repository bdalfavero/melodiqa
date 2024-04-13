#ifndef SAMPLER
#define SAMPLER

#include <random>
#include <cmath>
#include <mpi.h>
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

template <typename SpinSysT> /* Sublcass of spin system (e.g. Ising)*/
struct SamplerResult sample_spins_parallel(NeuralState nqstate, SpinSysT spin_sys, int nsweeps, int rank, int world_size) {
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

    int sweeps_this_rank = nsweeps / world_size;
    //std::cout << rank << " " << sweeps_this_rank << "sweeps" << std::endl;

    num_samples = 0;
    num_rejected = 0;
    for (int k = 0; k < sweeps_this_rank; k++) {
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

    //std::cout << avg_weight_gradient << std::endl;

    /* Get the total number of samples performed by all ranks. */
    int samples_all_ranks;
    MPI_Allreduce(&num_samples, &samples_all_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    /* Reduce the sums of gradients from all ranks. */
    std::complex<float> avg_local_energy_all = std::complex<float>(0., 0.);
    Eigen::MatrixXcf avg_weight_gradient_all = Eigen::MatrixXcf::Zero(nqstate.weights.rows(), nqstate.weights.cols());
    Eigen::VectorXcf avg_visible_gradient_all = Eigen::VectorXcf::Zero(nqstate.visible_bias.size());
    Eigen::VectorXcf avg_hidden_gradient_all = Eigen::VectorXcf::Zero(nqstate.hidden_bias.size());
    Eigen::MatrixXcf avg_le_wgrad_all = Eigen::MatrixXcf::Zero(nqstate.weights.rows(), nqstate.weights.cols());
    Eigen::VectorXcf avg_le_visgrad_all = Eigen::VectorXcf::Zero(nqstate.visible_bias.size());
    Eigen::VectorXcf avg_le_hidgrad_all = Eigen::VectorXcf::Zero(nqstate.hidden_bias.size());
    //std::cout << avg_weight_gradient_all << std::endl;
    MPI_Allreduce(&avg_local_energy, &avg_local_energy_all, 1, MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(avg_weight_gradient.data(), avg_weight_gradient_all.data(), avg_weight_gradient.size(), 
        MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    //std::cout << avg_weight_gradient_all << std::endl;
    MPI_Allreduce(avg_visible_gradient.data(), avg_visible_gradient_all.data(), avg_visible_gradient.size(), 
        MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(avg_hidden_gradient.data(), avg_hidden_gradient_all.data(), avg_hidden_gradient.size(), 
        MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(avg_le_wgrad.data(), avg_le_wgrad_all.data(), avg_le_wgrad.size(), 
        MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(avg_le_visgrad.data(), avg_le_visgrad_all.data(), avg_le_visgrad.size(), 
        MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(avg_le_hidgrad.data(), avg_le_hidgrad_all.data(), avg_le_hidgrad.size(), 
        MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    /* Divide averages by the number of samples for all ranks. */
    avg_local_energy_all = avg_local_energy_all / (float)samples_all_ranks;
    avg_weight_gradient_all = avg_weight_gradient_all / (float)samples_all_ranks;
    avg_visible_gradient_all = avg_visible_gradient_all / (float)samples_all_ranks;
    avg_hidden_gradient_all = avg_hidden_gradient_all / (float)samples_all_ranks;
    avg_le_wgrad_all = avg_le_wgrad_all / (float)samples_all_ranks;
    avg_le_visgrad_all = avg_le_visgrad_all / (float)samples_all_ranks;
    avg_le_hidgrad_all = avg_le_hidgrad_all / (float)samples_all_ranks;

    /* Compute the gradient form the collected samples. */
    weight_grad = 2.0 * (avg_le_wgrad_all - avg_local_energy_all * avg_weight_gradient_all);
    visible_grad = 2.0 * (avg_le_visgrad_all - avg_local_energy_all * avg_visible_gradient_all);
    hidden_grad = 2.0 * (avg_le_hidgrad_all - avg_local_energy_all * avg_hidden_gradient_all);

    SamplerResult result;
    result.num_samples = num_samples;
    result.num_rejected = num_rejected;
    result.avg_local_energy = avg_local_energy_all;
    result.weight_gradient = weight_grad;
    result.visible_gradient = visible_grad;
    result.hidden_gradient = hidden_grad;
    return result;
}
#endif