#ifndef TRAINING
#define TRAINING

#include "neural_state.hpp"
#include "spin_system.hpp"
#include "sampler.hpp"

struct training_step_result {
    float weight_grad_norm;
    float visible_grad_norm;
    float hidden_grad_norm;
    float energy;
};

template <typename T> /* Sublcass of spin system. */
struct training_step_result training_step(NeuralState &nqstate, T spin_sys, int nsweeps, float gamma) {
    SamplerResult sampler_result;
    // Norms of the total estimated gradients.
    float norm_weight_grad;
    float norm_visible_grad;
    float norm_hidden_grad;

    sampler_result = sample_spins<T>(nqstate, spin_sys, nsweeps);

    // Compute the estimated network gradient.
    norm_weight_grad = sampler_result.weight_gradient.norm();
    norm_visible_grad = sampler_result.visible_gradient.norm();
    norm_hidden_grad = sampler_result.hidden_gradient.norm();

    nqstate.weights -= std::complex<float>(gamma, 0.0) * sampler_result.weight_gradient;
    nqstate.visible_bias -= std::complex<float>(gamma, 0.0) * sampler_result.visible_gradient;
    nqstate.hidden_bias -= std::complex<float>(gamma, 0.0) * sampler_result.hidden_gradient;

    struct training_step_result result;
    result.weight_grad_norm = norm_weight_grad;
    result.visible_grad_norm = norm_visible_grad;
    result.hidden_grad_norm = norm_hidden_grad;
    result.energy = sampler_result.avg_local_energy.real();
    return result;
}

#endif