#ifndef NEURALSTATE
#define NEURALSTATE

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <vector>
#include <tuple>
#include <random>

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