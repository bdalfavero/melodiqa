#ifndef NEURALSTATE
#define NEURALSTATE

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <vector>
#include <random>

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

    std::vector<Eigen::VectorXcf> sample_spins(int nsweeps) {
        Eigen::VectorXcf spins = Eigen::VectorXcf::Constant(this->num_visible, 1, 1.0);
        std::vector<Eigen::VectorXcf> spin_configs;

        std::complex<float> old_psi, new_psi;
        float r, p;
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
                }
                spin_configs.push_back(spins);
            }
        }
        return spin_configs;
    }
};

#endif