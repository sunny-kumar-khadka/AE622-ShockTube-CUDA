#include "shock_tube.h"
#include <cassert>
#include <iomanip>

// Constructor
ShockTube::ShockTube(int n_grids) 
    : n_grids_(n_grids), h_(0.0), length_(1.0), t_(0.0), tau_(0.0), c_max_(0.0) {
    allocateHostMemory();
    allocateDeviceMemory();
    initializeHostMemory();
    initializeDeviceMemory();
}

// Destructor
ShockTube::~ShockTube() {
    freeHostMemory();
    freeDeviceMemory();
}

// CUDA error checking
void ShockTube::checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " 
                  << cudaGetErrorString(error) << std::endl;
        assert(false);
    }
}

void ShockTube::checkCudaError(const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " 
                  << cudaGetErrorString(error) << std::endl;
        assert(false);
    }
}

// Memory allocation
void ShockTube::allocateHostMemory() {
    u1_.resize(n_grids_);
    u2_.resize(n_grids_);
    u3_.resize(n_grids_);
    f1_.resize(n_grids_);
    f2_.resize(n_grids_);
    f3_.resize(n_grids_);
    vol_.resize(n_grids_);
    u1_temp_.resize(n_grids_);
    u2_temp_.resize(n_grids_);
    u3_temp_.resize(n_grids_);
}

void ShockTube::allocateDeviceMemory() {
    const size_t size = n_grids_ * sizeof(double);
    
    // Basic arrays
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_u1, size));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_u2, size));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_u3, size));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_f1, size));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_f2, size));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_f3, size));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_vol, size));
    
    // Scalar parameters
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_h, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_length, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_gamma, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_cfl, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_nu, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_tau, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_c_max, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_t, sizeof(double)));
    
    // Lax-Wendroff temporary arrays
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_u1_temp, size));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_u2_temp, size));
    CUDA_CHECK(cudaMalloc(&device_arrays_.d_u3_temp, size));
}

void ShockTube::freeHostMemory() {
    // Vectors automatically clean up
}

void ShockTube::freeDeviceMemory() {
    cudaFree(device_arrays_.d_u1);
    cudaFree(device_arrays_.d_u2);
    cudaFree(device_arrays_.d_u3);
    cudaFree(device_arrays_.d_f1);
    cudaFree(device_arrays_.d_f2);
    cudaFree(device_arrays_.d_f3);
    cudaFree(device_arrays_.d_vol);
    cudaFree(device_arrays_.d_h);
    cudaFree(device_arrays_.d_length);
    cudaFree(device_arrays_.d_gamma);
    cudaFree(device_arrays_.d_cfl);
    cudaFree(device_arrays_.d_nu);
    cudaFree(device_arrays_.d_tau);
    cudaFree(device_arrays_.d_c_max);
    cudaFree(device_arrays_.d_t);
    cudaFree(device_arrays_.d_u1_temp);
    cudaFree(device_arrays_.d_u2_temp);
    cudaFree(device_arrays_.d_u3_temp);
}

// Initialization
void ShockTube::initializeHostMemory() {
    length_ = 1.0;
    h_ = length_ / (n_grids_ - 1);
    t_ = 0.0;
    
    // Initialize with Sod shock tube problem
    for (int i = 0; i < n_grids_; ++i) {
        double rho, p, u = 0.0;
        if (i >= n_grids_ / 2) {
            rho = 0.125;
            p = 0.1;
        } else {
            rho = 1.0;
            p = 1.0;
        }
        
        double e = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
        u1_[i] = rho;
        u2_[i] = rho * u;
        u3_[i] = e;
        vol_[i] = 1.0;
    }
    
    updateHostCMax();
    tau_ = CFL * h_ / c_max_;
}

void ShockTube::initializeDeviceMemory() {
    // Copy scalar parameters to device
    CUDA_CHECK(cudaMemcpy(device_arrays_.d_h, &h_, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_arrays_.d_length, &length_, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_arrays_.d_gamma, &GAMMA, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_arrays_.d_cfl, &CFL, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_arrays_.d_nu, &EPS, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_arrays_.d_tau, &tau_, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_arrays_.d_t, &t_, sizeof(double), cudaMemcpyHostToDevice));
    
    // Initialize device arrays
    initDeviceMemory<<<1, 32>>>(n_grids_, device_arrays_.d_u1, device_arrays_.d_u2, device_arrays_.d_u3,
                                device_arrays_.d_vol, device_arrays_.d_h, device_arrays_.d_length,
                                device_arrays_.d_gamma, device_arrays_.d_cfl, device_arrays_.d_nu, device_arrays_.d_tau,
                                device_arrays_.d_c_max, device_arrays_.d_t);
    CUDA_CHECK_LAST();
}

void ShockTube::copyHostToDevice() {
    const size_t size = n_grids_ * sizeof(double);
    CUDA_CHECK(cudaMemcpy(device_arrays_.d_u1, u1_.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_arrays_.d_u2, u2_.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_arrays_.d_u3, u3_.data(), size, cudaMemcpyHostToDevice));
}

void ShockTube::copyDeviceToHost() {
    const size_t size = n_grids_ * sizeof(double);
    CUDA_CHECK(cudaMemcpy(u1_.data(), device_arrays_.d_u1, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(u2_.data(), device_arrays_.d_u2, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(u3_.data(), device_arrays_.d_u3, size, cudaMemcpyDeviceToHost));
}

// Boundary conditions
void ShockTube::applyHostBoundaryConditions() {
    u1_[0] = u1_[1];
    u2_[0] = -u2_[1];
    u3_[0] = u3_[1];
    u1_[n_grids_ - 1] = u1_[n_grids_ - 2];
    u2_[n_grids_ - 1] = -u2_[n_grids_ - 2];
    u3_[n_grids_ - 1] = u3_[n_grids_ - 2];
}

void ShockTube::applyDeviceBoundaryConditions() {
    boundaryCondition<<<1, 32>>>(n_grids_, device_arrays_.d_u1, device_arrays_.d_u2, device_arrays_.d_u3);
    CUDA_CHECK_LAST();
}

// Time step calculation
void ShockTube::updateHostTimeStep() {
    updateHostCMax();
    tau_ = CFL * h_ / c_max_;
}

void ShockTube::updateDeviceTimeStep() {
    updateTau<<<1, 1>>>(n_grids_, device_arrays_.d_u1, device_arrays_.d_u2, device_arrays_.d_u3,
                       device_arrays_.d_gamma, device_arrays_.d_c_max, device_arrays_.d_h,
                       device_arrays_.d_cfl, device_arrays_.d_tau);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(&tau_, device_arrays_.d_tau, sizeof(double), cudaMemcpyDeviceToHost));
}

// Utility functions
void ShockTube::updateHostCMax() {
    c_max_ = 0.0;
    for (int i = 0; i < n_grids_; ++i) {
        if (u1_[i] == 0.0) continue;
        
        double rho = u1_[i];
        double u = u2_[i] / rho;
        double p = (u3_[i] - 0.5 * rho * u * u) * (GAMMA - 1.0);
        double c = std::sqrt(GAMMA * std::abs(p) / rho);
        c_max_ = std::max(c_max_, c + std::abs(u));
    }
}

void ShockTube::updateHostAverages() {
    double rho_avg = 0.0, u_avg = 0.0, e_avg = 0.0, p_avg = 0.0;
    
    for (int i = 0; i < n_grids_; ++i) {
        double rho = u1_[i];
        double u = u2_[i] / rho;
        double e = u3_[i];
        double p = (u3_[i] - 0.5 * rho * u * u) * (GAMMA - 1.0);
        
        rho_avg += rho;
        u_avg += u;
        e_avg += e;
        p_avg += p;
    }
    
    rho_avg /= n_grids_;
    u_avg /= n_grids_;
    e_avg /= n_grids_;
    p_avg /= n_grids_;
}

// Write solution to file
void ShockTube::writeToFile(const std::string& filename) const {
    std::string full_path = "../results/" + filename;
    std::ofstream file(full_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << full_path << std::endl;
        return;
    }
    
    file << "variables = position, density, velocity, pressure, momentum, energy, totalEnergy, temperature, soundVelocity, machNumber, enthalpy" << std::endl;
    
    for (int i = 0; i < n_grids_; ++i) {
        double rho = u1_[i];
        double u = u2_[i] / rho;
        double p = (u3_[i] - 0.5 * rho * u * u) * (GAMMA - 1.0);
        double m = u2_[i];
        double e = u3_[i];
        double E = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
        double T = p / rho;
        double c = std::sqrt(GAMMA * p / rho);
        double M = u / c;
        double h = e + p / rho;
        double x = static_cast<double>(i) / static_cast<double>(n_grids_);
        
        file << std::scientific << std::setprecision(15)
             << x << " " << rho << " " << u << " " << p << " " << m << " " << e << " " << E
             << " " << T << " " << c << " " << M << " " << h << std::endl;
    }
    
    file.close();
}

// Main simulation interface
void ShockTube::runSimulation(double t_max, 
                             const std::string& output_prefix, int n_output) {
    std::cout << "Running Lax-Wendroff simulation with " << n_grids_ << " grid points..." << std::endl;
    
    // Run both host and device simulations
    std::cout << "Running both host and device simulations..." << std::endl;
    
    // Host simulation
    std::string host_filename = output_prefix + "_lax_host.dat";
    runHostSimulation(t_max, host_filename);
    
    // Device simulation
    runDeviceSimulation(t_max, output_prefix + "_lax_device", n_output);
}

void ShockTube::runHostSimulation(double t_max, const std::string& filename) {
    std::cout << "Running host Lax-Wendroff simulation..." << std::endl;
    
    double t = 0.0;
    while (t < t_max - EPS) {
        applyHostBoundaryConditions();
        updateHostTimeStep();
        
        if (t + tau_ > t_max) {
            tau_ = t_max - t;
        }
        
        hostLaxWendroffStep();
        t += tau_;
    }
    
    writeToFile(filename);
    std::cout << "Solution written to ../results/" << filename << std::endl;
}

void ShockTube::runDeviceSimulation(double t_max, 
                                   const std::string& output_prefix, int n_output) {
    std::cout << "Running device Lax-Wendroff simulation..." << std::endl;
    
    // Write initial condition
    writeToFile(output_prefix + "00.dat");
    
    double t_increment = t_max / n_output;
    for (int step = 1; step <= n_output; ++step) {
        double t_target = step * t_increment;
        
        // Reinitialize for each time step
        initializeDeviceMemory();
        
        double t = 0.0;
        while (t < t_target - EPS) {
            updateDeviceTimeStep();
            
            if (t + tau_ > t_target) {
                tau_ = t_target - t;
                CUDA_CHECK(cudaMemcpy(device_arrays_.d_tau, &tau_, sizeof(double), cudaMemcpyHostToDevice));
            }
            
            deviceLaxWendroffStep();
            t += tau_;
        }
        
        copyDeviceToHost();
        
        std::string filename = output_prefix + (step < 10 ? "0" : "") + std::to_string(step) + ".dat";
        writeToFile(filename);
        std::cout << "Solution written to ../results/" << filename << std::endl;
    }
}

// Host Lax-Wendroff scheme
void ShockTube::hostLaxWendroffStep() {
    // Update fluxes
    for (int j = 0; j < n_grids_; ++j) {
        double rho = u1_[j];
        double m = u2_[j];
        double e = u3_[j];
        double p = (GAMMA - 1.0) * (e - 0.5 * m * m / rho);
        
        f1_[j] = m;
        f2_[j] = m * m / rho + p;
        f3_[j] = m / rho * (e + p);
    }
    
    // Half step
    for (int j = 1; j < n_grids_ - 1; ++j) {
        u1_temp_[j] = 0.5 * (u1_[j + 1] + u1_[j]) - 0.5 * tau_ / h_ * (f1_[j + 1] - f1_[j]);
        u2_temp_[j] = 0.5 * (u2_[j + 1] + u2_[j]) - 0.5 * tau_ / h_ * (f2_[j + 1] - f2_[j]);
        u3_temp_[j] = 0.5 * (u3_[j + 1] + u3_[j]) - 0.5 * tau_ / h_ * (f3_[j + 1] - f3_[j]);
    }
    
    // Apply boundary conditions to temporary arrays
    u1_temp_[0] = u1_temp_[1];
    u2_temp_[0] = -u2_temp_[1];
    u3_temp_[0] = u3_temp_[1];
    u1_temp_[n_grids_ - 1] = u1_temp_[n_grids_ - 2];
    u2_temp_[n_grids_ - 1] = -u2_temp_[n_grids_ - 2];
    u3_temp_[n_grids_ - 1] = u3_temp_[n_grids_ - 2];
    
    // Update fluxes for temporary arrays
    for (int j = 0; j < n_grids_; ++j) {
        double rho = u1_temp_[j];
        double m = u2_temp_[j];
        double e = u3_temp_[j];
        double p = (GAMMA - 1.0) * (e - 0.5 * m * m / rho);
        
        f1_[j] = m;
        f2_[j] = m * m / rho + p;
        f3_[j] = m / rho * (e + p);
    }
    
    // Full step
    for (int j = 1; j < n_grids_ - 1; ++j) {
        u1_temp_[j] = u1_[j] - tau_ / h_ * (f1_[j] - f1_[j - 1]);
        u2_temp_[j] = u2_[j] - tau_ / h_ * (f2_[j] - f2_[j - 1]);
        u3_temp_[j] = u3_[j] - tau_ / h_ * (f3_[j] - f3_[j - 1]);
    }
    
    // Update solution
    for (int j = 1; j < n_grids_ - 1; ++j) {
        u1_[j] = u1_temp_[j];
        u2_[j] = u2_temp_[j];
        u3_[j] = u3_temp_[j];
    }
}

// Device Lax-Wendroff scheme
void ShockTube::deviceLaxWendroffStep() {
    laxWendroffStep<<<1, 32>>>(n_grids_, device_arrays_.d_u1, device_arrays_.d_u2, device_arrays_.d_u3,
                               device_arrays_.d_u1_temp, device_arrays_.d_u2_temp, device_arrays_.d_u3_temp,
                               device_arrays_.d_f1, device_arrays_.d_f2, device_arrays_.d_f3,
                               device_arrays_.d_tau, device_arrays_.d_h, device_arrays_.d_gamma);
    CUDA_CHECK_LAST();
}

// Test functions
void ShockTube::runTests() {
    std::cout << "Running unit tests..." << std::endl;
    runHostTests();
    runDeviceTests();
}

void ShockTube::runHostTests() {
    std::cout << "Host tests:" << std::endl;
    // Test implementations would go here
}

void ShockTube::runDeviceTests() {
    std::cout << "Device tests:" << std::endl;
    // Test implementations would go here
}
