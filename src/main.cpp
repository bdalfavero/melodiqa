#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "../include/neural_state.hpp"
#include "../include/spin_system.hpp"

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
    
    Eigen::VectorXcf spins = Eigen::VectorXcf::Constant(num_visible, 1, 1.0);
    //std::cout << spins << std::endl;

    std::complex<float> s_psi = nqstate.evalute_state(spins);
    std::cout << "<s|psi> = " << s_psi << std::endl;

    struct nqs_sweep_result result = nqstate.sample_spins(5);
    std::vector<Eigen::VectorXcf> spin_configs = result.spin_configs;
    for (int i = 0; i < spin_configs.size(); i++) {
        //std::cout << spins_to_bits(spin_configs[i]) << '\n';
    }
    std::cout << "number rejected = " << result.num_rejected << std::endl;
    std::cout << "number of samples = " << result.num_samples << std::endl;

    struct nqs_gradient nqs_grad = nqstate.gradient(spins);
    std::cout << "Weight gradient\n" << nqs_grad.weight_grad << std::endl;
    std::cout << "Visible gradient\n" << nqs_grad.visible_grad << std::endl;
    std::cout << "Hidden gradient\n" << nqs_grad.hidden_grad << std::endl;

    IsingSystem ising = IsingSystem(num_visible, 1.0, 0.0);
    std::cout << "Number of spins = " << ising.size << std::endl;
    std::cout << "J = " << ising.J << std::endl;
    std::cout << "B = " << ising.B << std::endl;
    
    return 0;
}