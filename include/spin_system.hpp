#ifndef SPINSYS
#define SPINSYS

#include "neural_state.hpp"
#include <iostream>

class SpinSystem {
public:
    int size;

    SpinSystem(int size) {
        this->size = size;
    }

    virtual std::complex<float> local_energy(NeuralState nqstate, Eigen::VectorXcf spins) {
        std::cerr << "Local energy is not implemented by SpinSystem." << std::endl;
        exit(-1);
        return std::complex<float>(-0.0, 0.0);
    } 
};

class IsingSystem: public SpinSystem {
public:
    float J, B; // J is interaction term, B is on-site magnetic field.

    IsingSystem(int size, float J, float B): SpinSystem(size) {
        this->size = size;
        this->J = J;
        this->B = B;
    }

    std::complex<float> local_energy(NeuralState nqstate, Eigen::VectorXcf spins) {
        //std::cout << "In local energy" << std::endl;
        std::complex<float> s_psi = nqstate.evalute_state(spins);
        std::complex<float> local_energy = std::complex<float>(0.0, 0.0);
        for (int i = 0; i < nqstate.num_visible; i++) {
            local_energy += this->B * spins(i);
            if (i != 0) local_energy += this->J * spins(i - 1) * spins(i);
        }
        //std::cout << local_energy << std::endl;
        local_energy *= s_psi;
        return local_energy / s_psi;
    }
};

#endif