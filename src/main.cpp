#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <complex>
#include <fstream>
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
    std::ifstream input(argv[1]);

    int num_visible;
    int num_hidden;
    input >> num_visible;
    input >> num_hidden;
    NeuralState nqstate = NeuralState(num_visible, num_hidden);
    nqstate.visible_bias = Eigen::VectorXcf::Random(num_visible);
    nqstate.hidden_bias = Eigen::VectorXcf::Random(num_hidden);
    //nqstate.weights = Eigen::MatrixXcf::Identity(num_hidden, num_visible);
    nqstate.weights = Eigen::MatrixXcf::Random(num_hidden, num_visible);
    //std::cout << "Finished assignments." << std::endl;

    IsingSystem ising = IsingSystem(num_visible, 0.0, 1.0);

    int nsweeps;
    input >> nsweeps;
    float gamma;
    input >> gamma;
    std::cerr << gamma << std::endl;
    struct training_step_result result;
    std::cout << "i,weight_grad,visible_grad,hidden_grad" << std::endl;
    for (int i = 0; i < 1000; i++) {
        result = training_step<IsingSystem>(nqstate, ising, nsweeps, gamma);
        std::cout << i << "," << result.weight_grad_norm << "," 
                << result.visible_grad_norm << "," 
                << result.hidden_grad_norm << std::endl;;
    }

    return 0;
}