#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

/**
 * @brief CUDA implementation of Lax-Wendroff scheme to solve Sod's 1D Shock Tube Problem
 * 
 * This class implements numerical solutions to the 1D Euler equations using
 * the Lax-Wendroff scheme with CUDA acceleration.
 */
class ShockTube {
public:
    // Physical constants
    const double GAMMA = 1.4;           // Ratio of specific heats
    const double CFL = 0.9;             // Courant-Friedrichs-Lewy number
    const double EPS = 1e-14;           // Convergence tolerance

    /**
     * @brief Constructor
     * @param n_grids Number of grid points (default: 200)
     */
    explicit ShockTube(int n_grids = 200);
    
    /**
     * @brief Destructor - automatically cleans up memory
     */
    ~ShockTube();

    // Disable copy constructor and assignment operator
    ShockTube(const ShockTube&) = delete;
    ShockTube& operator=(const ShockTube&) = delete;

    /**
     * @brief Run the complete simulation
     * @param t_max Final simulation time
     * @param output_prefix Prefix for output files
     * @param n_output Number of output time steps
     */
    void runSimulation(double t_max = 0.2, 
                      const std::string& output_prefix = "solution",
                      int n_output = 21);

    /**
     * @brief Run host-based simulation
     * @param t_max Final simulation time
     * @param filename Output filename
     */
    void runHostSimulation(double t_max, const std::string& filename);

    /**
     * @brief Run device-based simulation
     * @param t_max Final simulation time
     * @param output_prefix Output file prefix
     * @param n_output Number of output time steps
     */
    void runDeviceSimulation(double t_max, 
                           const std::string& output_prefix, int n_output);

    /**
     * @brief Write current solution to file
     * @param filename Output filename
     */
    void writeToFile(const std::string& filename) const;

    /**
     * @brief Run all unit tests
     */
    void runTests();

private:
    // Grid parameters
    int n_grids_;
    double h_;              // Grid spacing
    double length_;         // Domain length
    double t_;              // Current time
    double tau_;            // Time step
    double c_max_;          // Maximum wave speed

    // Host arrays
    std::vector<double> u1_, u2_, u3_;  // Conservative variables (density, momentum, energy)
    std::vector<double> f1_, f2_, f3_;  // Fluxes
    std::vector<double> vol_;           // Cell volumes
    std::vector<double> u1_temp_, u2_temp_, u3_temp_;  // Temporary arrays for Lax-Wendroff

    // Device arrays (managed with smart pointers for automatic cleanup)
    struct DeviceArrays {
        double *d_u1, *d_u2, *d_u3;
        double *d_f1, *d_f2, *d_f3;
        double *d_vol;
        double *d_h, *d_length, *d_gamma, *d_cfl, *d_nu, *d_tau, *d_c_max, *d_t;
        double *d_u1_temp, *d_u2_temp, *d_u3_temp;
    } device_arrays_;

    // Memory management
    void allocateHostMemory();
    void allocateDeviceMemory();
    void freeHostMemory();
    void freeDeviceMemory();

    // Initialization
    void initializeHostMemory();
    void initializeDeviceMemory();
    void copyHostToDevice();
    void copyDeviceToHost();

    // Boundary conditions
    void applyHostBoundaryConditions();
    void applyDeviceBoundaryConditions();

    // Time step calculation
    void updateHostTimeStep();
    void updateDeviceTimeStep();

    // Lax-Wendroff scheme
    void hostLaxWendroffStep();
    void deviceLaxWendroffStep();

    // Utility functions
    void updateHostAverages();
    void updateHostCMax();
    void updateDeviceCMax();

    // CUDA error checking
    static void checkCudaError(cudaError_t error, const char* file, int line);
    static void checkCudaError(const char* file, int line);

    // Test functions
    void runHostTests();
    void runDeviceTests();
    bool testMemoryAllocation();
    bool testBoundaryConditions();
    bool testLaxWendroffStep();
};

// CUDA kernel declarations
__global__ void initDeviceMemory(int n_grids, double* d_u1, double* d_u2, double* d_u3, 
                                double* d_vol, double* d_h, double* d_length, double* d_gamma, 
                                double* d_cfl, double* d_nu, double* d_tau, double* d_c_max, double* d_t);

__global__ void boundaryCondition(int n_grids, double* d_u1, double* d_u2, double* d_u3);

__global__ void updateTau(int n_grids, const double* d_u1, const double* d_u2, const double* d_u3, 
                         const double* d_gamma, double* d_c_max, const double* d_h, 
                         const double* d_cfl, double* d_tau);

__global__ void laxWendroffStep(int n_grids, double* d_u1, double* d_u2, double* d_u3, 
                               double* d_u1_temp, double* d_u2_temp, double* d_u3_temp,
                               double* d_f1, double* d_f2, double* d_f3, 
                               const double* d_tau, const double* d_h, const double* d_gamma);

// Device helper functions
__device__ void updateCMax(int n_grids, const double* d_u1, const double* d_u2, const double* d_u3, 
                          const double* d_gamma, double* d_c_max);

__device__ void updateFlux(int n_grids, const double* d_u1, const double* d_u2, const double* d_u3,
                          double* d_f1, double* d_f2, double* d_f3, const double* d_gamma);

__device__ void halfStep(int n_grids, const double* d_u1, const double* d_u2, const double* d_u3,
                        double* d_u1_temp, double* d_u2_temp, double* d_u3_temp,
                        const double* d_f1, const double* d_f2, const double* d_f3, 
                        const double* d_tau, const double* d_h);

__device__ void step(int n_grids, const double* d_u1, const double* d_u2, const double* d_u3,
                    double* d_u1_temp, double* d_u2_temp, double* d_u3_temp,
                    const double* d_f1, const double* d_f2, const double* d_f3, 
                    const double* d_tau, const double* d_h);

__device__ void updateU(int n_grids, double* d_u1, double* d_u2, double* d_u3,
                       const double* d_u1_temp, const double* d_u2_temp, const double* d_u3_temp);

__device__ void deviceBoundaryCondition(int n_grids, double* d_u1, double* d_u2, double* d_u3);

// CUDA error checking macro
#define CUDA_CHECK(call) ShockTube::checkCudaError(call, __FILE__, __LINE__)
#define CUDA_CHECK_LAST() ShockTube::checkCudaError(__FILE__, __LINE__)
