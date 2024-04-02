#include <iostream>
#include <Eigen/Dense>
#include "../include/neural_state.hpp"

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
    //nqstate.visible_bias = Eigen::VectorXcf::Constant(num_visible, 1, 1.0);
    nqstate.visible_bias = Eigen::VectorXcf::Zero(num_visible);
    nqstate.hidden_bias = Eigen::VectorXcf::Zero(num_hidden);
    Eigen::VectorXcf spins = Eigen::VectorXcf::Constant(num_visible, 1, 1.0);
    //std::cout << spins << std::endl;

    std::complex<float> s_psi = nqstate.evalute_state(spins);
    std::cout << "<s|psi> = " << s_psi << std::endl;

    std::cout << spins_to_bits(spins) << std::endl;

    std::vector<Eigen::VectorXcf> spin_configs = nqstate.sample_spins(5);
    for (int i = 0; i < spin_configs.size(); i++) {
        std::cout << spins_to_bits(spin_configs[i]) << '\n';
    }
    
    return 0;
}