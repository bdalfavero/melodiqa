#include <Eigen/Dense>
#include <iostream>
#include <stdio.h>
#include "../include/neural_state.hpp"
#include "../include/spin_system.hpp"
#include "../include/training.hpp"

struct training_step_result training_step(NeuralState nqstate, SpinSystem spin_sys, int nsweeps, float gamma) {
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
    // Norms of the total estimated gradients.
    float norm_weight_grad;
    float norm_visible_grad;
    float norm_hidden_grad;

    // Compute averages of local energy, gradients, and le * grad.
    nqs_sweep_result sweep_result = nqstate.sample_spins(nsweeps);
    for (int i = 0; i < sweep_result.num_samples; i++) {
        //std::cout << "In training loop, i = " << i << " / " << sweep_result.num_samples << "\n";
        spins = sweep_result.spin_configs[i];
        s_psi = nqstate.evalute_state(spins);
        grad = nqstate.gradient(spins);
        local_energy = spin_sys.local_energy(nqstate, spins);
        avg_local_energy += local_energy;
        avg_weight_gradient += grad.weight_grad;
        avg_visible_gradient += grad.visible_grad;
        avg_hidden_gradient += grad.hidden_grad;
        //std::cout << "Finished average gradients." << std::endl;
        avg_le_wgrad += local_energy * grad.weight_grad;
        avg_le_visgrad += local_energy * grad.visible_grad;
        avg_le_hidgrad += local_energy * grad.hidden_grad;
        //std::cout << "Finished local energy * average gradients." << std::endl;
    }
    avg_local_energy = avg_local_energy / (float)sweep_result.num_samples;
    avg_weight_gradient = avg_weight_gradient / (float)sweep_result.num_samples;
    avg_visible_gradient = avg_visible_gradient / (float)sweep_result.num_samples;
    avg_hidden_gradient = avg_hidden_gradient / (float)sweep_result.num_samples;
    avg_le_wgrad = avg_le_wgrad / (float)sweep_result.num_samples;
    avg_le_visgrad = avg_le_visgrad / (float)sweep_result.num_samples;
    avg_le_hidgrad = avg_le_hidgrad / (float)sweep_result.num_samples;

    // Compute the estimated network gradient.
    //std::cout << "Comuting total gradients." << std::endl;
    weight_grad = 2.0 * (avg_le_wgrad - avg_local_energy * avg_weight_gradient);
    visible_grad = 2.0 * (avg_le_visgrad - avg_local_energy * avg_visible_gradient);
    hidden_grad = 2.0 * (avg_le_hidgrad - avg_local_energy * avg_hidden_gradient);
    norm_weight_grad = weight_grad.norm();
    norm_visible_grad = visible_grad.norm();
    norm_hidden_grad = hidden_grad.norm();

    printf("%f %f %f\n", norm_weight_grad, norm_visible_grad, norm_hidden_grad);

    nqstate.weights -= gamma * weight_grad;
    nqstate.visible_bias -= gamma * visible_grad;
    nqstate.hidden_bias -= gamma * hidden_grad;

    struct training_step_result result;
    result.weight_grad_norm = norm_weight_grad;
    result.visible_grad_norm = norm_visible_grad;
    result.hidden_grad_norm = norm_hidden_grad;
    result.energy = avg_local_energy.real();
    return result;
}

