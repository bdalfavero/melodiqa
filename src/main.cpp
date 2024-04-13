#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <complex>
#include <fstream>
#include <mpi.h>
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

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
    int num_steps;
    input >> num_steps;
    struct training_step_result result;
    if (rank == 0) std::cout << "i,weight_grad,visible_grad,hidden_grad,energy" << std::endl;
    Eigen::MatrixXcf old_weights;
    Eigen::VectorXcf old_vis_bias;
    Eigen::VectorXcf old_hid_bias;
    for (int i = 0; i < num_steps; i++) {
        old_weights = nqstate.weights;
        old_vis_bias = nqstate.visible_bias;
        old_hid_bias = nqstate.hidden_bias;
        result = training_step_parallel<IsingSystem>(nqstate, ising, nsweeps, gamma, rank, world_size);
        //result = training_step<IsingSystem>(nqstate, ising, nsweeps, gamma);
        if (rank == 0) {
            std::cout << i << "," << result.weight_grad_norm << "," 
                    << result.visible_grad_norm << "," 
                    << result.hidden_grad_norm << ","
                    << result.energy << std::endl;
        }
        //std::cerr << (old_weights - nqstate.weights).norm() << std::endl;
    }

    MPI_Finalize();

    return 0;
}