#ifndef NEURALSTATE
#define NEURALSTATE

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <vector>
#include <tuple>
#include <random>

struct nqs_sweep_result {
    std::vector<Eigen::VectorXcf> spin_configs;
    int num_samples;
    int num_rejected;
};

struct nqs_gradient {
    Eigen::MatrixXcf weight_grad;
    Eigen::VectorXcf visible_grad;
    Eigen::VectorXcf hidden_grad;
};

class NeuralState {
public:
    int num_hidden;
    int num_visible;
    Eigen::MatrixXcf weights;
    Eigen::VectorXcf hidden_bias;
    Eigen::VectorXcf visible_bias;

    NeuralState(int num_visible, int num_hidden) {
        this->num_hidden = num_hidden;
        this->num_visible = num_visible;
        this->weights = Eigen::MatrixXcf::Zero(num_hidden, num_visible);
        this->hidden_bias = Eigen::VectorXcf::Zero(num_hidden);
        this->visible_bias = Eigen::VectorXcf::Zero(num_visible);
    }

    std::complex<float> evalute_state(Eigen::VectorXcf spins) {
        /* Evaluate the wave function at a spin configuration. */
        Eigen::VectorXcf theta = this->hidden_bias + this->weights * spins;
        Eigen::VectorXcf f_i = Eigen::VectorXcf::Zero(this->num_visible);
        for (int i = 0; i < this->num_visible; i++) {
            f_i(i) = std::complex<float>(2.0, 0.0) * cosh(theta(i));
        }
        std::complex<float> a_dot_s = spins.dot(this->visible_bias);
        return exp(a_dot_s) * f_i.prod();
    }

    /* TODO: Modify this to have a callback to accumulate gradients/observables
     * The callback should take three arguments: The NQS, A spin config, and an object of unspecified
     * type. The third argument is the quantity that we are adding up, like the grads or 
     * observables. It should be a dummy argument so that we just update it by calling
     * the callback function.
     * The "host function" that calls the sweep method will divide by the number of samples.
     */
    struct nqs_sweep_result sample_spins(int nsweeps) {
        Eigen::VectorXcf spins = Eigen::VectorXcf::Constant(this->num_visible, 1, 1.0);
        std::vector<Eigen::VectorXcf> spin_configs;
        std::complex<float> old_psi, new_psi;
        float r, p;
        int num_samples, num_rejected;

        num_samples = 0;
        num_rejected = 0;
        for (int k = 0; k < nsweeps; k++) {
            for (int i = 0; i < this->num_visible; i++) {
                old_psi = this->evalute_state(spins);
                // Flip the i^th spin.
                spins(i) = std::complex<float>(-1.0, 0.0) * spins(i);
                new_psi = this->evalute_state(spins);
                p = pow(std::abs(new_psi / old_psi), 2);
                r = (float)rand() / (float)RAND_MAX;
                if (r >= p) {
                    // Flip rejected.
                    spins(i) = std::complex<float>(-1.0, 0.0) * spins(i);
                    num_rejected++;
                }
                spin_configs.push_back(spins);
                num_samples++;
            }
        }
        struct nqs_sweep_result result;
        result.spin_configs = spin_configs;
        result.num_rejected = num_rejected;
        result.num_samples = num_samples;
        return result;
    }

    struct nqs_gradient gradient(Eigen::VectorXcf spins) {
        // N.b. This is the gradient divided by the wave function value!
        Eigen::VectorXcf theta = this->hidden_bias + this->weights * spins;
        Eigen::VectorXcf visible_gradient = spins;
        Eigen::VectorXcf hidden_gradient = Eigen::VectorXcf::Zero(this->hidden_bias.size());
        Eigen::MatrixXcf weight_gradient = Eigen::MatrixXcf::Zero(this->weights.rows(), this->weights.cols());
        for (int j = 0; j < this->num_hidden; j++) {
            hidden_gradient(j) = tanh(theta(j));
            for (int i = 0; i < this->num_visible; i++) {
                weight_gradient(j, i) = spins(i) * tanh(theta(j));
            }
        }
        struct nqs_gradient nqs_grad;
        nqs_grad.weight_grad = weight_gradient;
        nqs_grad.visible_grad = visible_gradient;
        nqs_grad.hidden_grad = hidden_gradient;
        return nqs_grad;
    }
};

#endif