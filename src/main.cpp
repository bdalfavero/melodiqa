#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "../include/neural_state.hpp"
#include "../include/spin_system.hpp"
#include "../include/training.hpp"

std::string spins_to_bits(Eigen::VectorXcf spins) {
    std::string bits = "";
    for (int i = 0; i < spins.rows(); i++) {
        if (spins(i).real() > 0.0) {
            bits += '1';
        } else {
            bits += '0';
        }
    }
    return bits;
}

int main(int argc, char *argv[]) {
    int num_visible = 4;
    int num_hidden = 5;
    NeuralState nqstate = NeuralState(num_visible, num_hidden);
    nqstate.visible_bias = Eigen::VectorXcf::Constant(num_visible, 1, 1.0);
    nqstate.hidden_bias = Eigen::VectorXcf::Constant(num_hidden, 1, 1.0);
    //nqstate.weights = Eigen::MatrixXcf::Identity(num_hidden, num_visible);
    nqstate.weights = Eigen::MatrixXcf::Zero(num_hidden, num_visible);
    //std::cout << "Finished assignments." << std::endl;

    IsingSystem ising = IsingSystem(num_visible, 1.0, 0.0);

    int nsweeps = 100;
    float gamma = 1e-4;
    struct training_step_result result; 
    for (int i = 0; i < 10; i++) {
        result = training_step(nqstate, ising, nsweeps, gamma);
        std::cout << "weight grad norm = " << std::setprecision(6) << result.weight_grad_norm << std::endl;
        std::cout << "visible grad norm = " << std::setprecision(6) << result.visible_grad_norm << std::endl;
        std::cout << "hidden grad norm = " << std::setprecision(6) << result.hidden_grad_norm << std::endl;
    }

    return 0;
}