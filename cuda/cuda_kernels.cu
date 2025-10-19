#include "shock_tube.h"

// Device helper function to update maximum wave speed
__device__ void updateCMax(int n_grids, const double* d_u1, const double* d_u2, const double* d_u3, 
                          const double* d_gamma, double* d_c_max) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double local_max = 0.0;
    
    for (int i = index; i < n_grids; i += stride) {
        if (d_u1[i] == 0.0) continue;
        
        double rho = d_u1[i];
        double u = d_u2[i] / rho;
        double p = (d_u3[i] - 0.5 * rho * u * u) * (*d_gamma - 1.0);
        double c = sqrt(*d_gamma * fabs(p) / rho);
        local_max = fmax(local_max, c + fabs(u));
    }
    
    // Use shared memory for reduction
    __shared__ double shared_max;
    if (threadIdx.x == 0) shared_max = 0.0;
    __syncthreads();
    
    atomicMax((unsigned long long*)&shared_max, __double_as_longlong(local_max));
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicMax((unsigned long long*)d_c_max, __double_as_longlong(shared_max));
    }
}

// Initialize device memory with Sod shock tube problem
__global__ void initDeviceMemory(int n_grids, double* d_u1, double* d_u2, double* d_u3, 
                                double* d_vol, double* d_h, double* d_length, double* d_gamma, 
                                double* d_cfl, double* d_nu, double* d_tau, double* d_c_max, double* d_t) {
    *d_t = 0.0;
    *d_length = 1.0;
    *d_gamma = 1.4;
    *d_cfl = 0.9;
    *d_nu = 0.0;
    *d_h = *d_length / (n_grids - 1);
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n_grids; i += stride) {
        double rho, p, u = 0.0;
        if (i >= n_grids / 2) {
            rho = 0.125;
            p = 0.1;
        } else {
            rho = 1.0;
            p = 1.0;
        }
        
        double e = p / (*d_gamma - 1.0) + 0.5 * rho * u * u;
        d_u1[i] = rho;
        d_u2[i] = rho * u;
        d_u3[i] = e;
        d_vol[i] = 1.0;
    }
    
    updateCMax(n_grids, d_u1, d_u2, d_u3, d_gamma, d_c_max);
    *d_tau = (*d_cfl) * (*d_h) / (*d_c_max);
}

// Apply boundary conditions
__global__ void boundaryCondition(int n_grids, double* d_u1, double* d_u2, double* d_u3) {
    d_u1[0] = d_u1[1];
    d_u2[0] = -d_u2[1];
    d_u3[0] = d_u3[1];
    d_u1[n_grids - 1] = d_u1[n_grids - 2];
    d_u2[n_grids - 1] = -d_u2[n_grids - 2];
    d_u3[n_grids - 1] = d_u3[n_grids - 2];
}

// Update time step
__global__ void updateTau(int n_grids, const double* d_u1, const double* d_u2, const double* d_u3, 
                         const double* d_gamma, double* d_c_max, const double* d_h, 
                         const double* d_cfl, double* d_tau) {
    updateCMax(n_grids, d_u1, d_u2, d_u3, d_gamma, d_c_max);
    *d_tau = *d_cfl * *d_h / *d_c_max;
}

// Device boundary condition helper
__device__ void deviceBoundaryCondition(int n_grids, double* d_u1, double* d_u2, double* d_u3) {
    d_u1[0] = d_u1[1];
    d_u2[0] = -d_u2[1];
    d_u3[0] = d_u3[1];
    d_u1[n_grids - 1] = d_u1[n_grids - 2];
    d_u2[n_grids - 1] = -d_u2[n_grids - 2];
    d_u3[n_grids - 1] = d_u3[n_grids - 2];
}

// Update fluxes
__device__ void updateFlux(int n_grids, const double* d_u1, const double* d_u2, const double* d_u3,
                          double* d_f1, double* d_f2, double* d_f3, const double* d_gamma) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n_grids; i += stride) {
        double rho = d_u1[i];
        double m = d_u2[i];
        double e = d_u3[i];
        double p = (*d_gamma - 1.0) * (e - 0.5 * m * m / rho);
        
        d_f1[i] = m;
        d_f2[i] = m * m / rho + p;
        d_f3[i] = m / rho * (e + p);
    }
}

// Half step for Lax-Wendroff
__device__ void halfStep(int n_grids, const double* d_u1, const double* d_u2, const double* d_u3,
                        double* d_u1_temp, double* d_u2_temp, double* d_u3_temp,
                        const double* d_f1, const double* d_f2, const double* d_f3, 
                        const double* d_tau, const double* d_h) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n_grids; i += stride) {
        if (i > 0 && i < n_grids - 1) {
            d_u1_temp[i] = 0.5 * (d_u1[i + 1] + d_u1[i]) - 0.5 * (*d_tau) / (*d_h) * (d_f1[i + 1] - d_f1[i]);
            d_u2_temp[i] = 0.5 * (d_u2[i + 1] + d_u2[i]) - 0.5 * (*d_tau) / (*d_h) * (d_f2[i + 1] - d_f2[i]);
            d_u3_temp[i] = 0.5 * (d_u3[i + 1] + d_u3[i]) - 0.5 * (*d_tau) / (*d_h) * (d_f3[i + 1] - d_f3[i]);
        }
    }
}

// Full step for Lax-Wendroff
__device__ void step(int n_grids, const double* d_u1, const double* d_u2, const double* d_u3,
                    double* d_u1_temp, double* d_u2_temp, double* d_u3_temp,
                    const double* d_f1, const double* d_f2, const double* d_f3, 
                    const double* d_tau, const double* d_h) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n_grids; i += stride) {
        if (i > 0 && i < n_grids - 1) {
            d_u1_temp[i] = d_u1[i] - (*d_tau) / (*d_h) * (d_f1[i] - d_f1[i - 1]);
            d_u2_temp[i] = d_u2[i] - (*d_tau) / (*d_h) * (d_f2[i] - d_f2[i - 1]);
            d_u3_temp[i] = d_u3[i] - (*d_tau) / (*d_h) * (d_f3[i] - d_f3[i - 1]);
        }
    }
}

// Update solution
__device__ void updateU(int n_grids, double* d_u1, double* d_u2, double* d_u3,
                       const double* d_u1_temp, const double* d_u2_temp, const double* d_u3_temp) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n_grids; i += stride) {
        if (i > 0 && i < n_grids - 1) {
            d_u1[i] = d_u1_temp[i];
            d_u2[i] = d_u2_temp[i];
            d_u3[i] = d_u3_temp[i];
        }
    }
}

// Lax-Wendroff step kernel
__global__ void laxWendroffStep(int n_grids, double* d_u1, double* d_u2, double* d_u3, 
                               double* d_u1_temp, double* d_u2_temp, double* d_u3_temp,
                               double* d_f1, double* d_f2, double* d_f3, 
                               const double* d_tau, const double* d_h, const double* d_gamma) {
    updateFlux(n_grids, d_u1, d_u2, d_u3, d_f1, d_f2, d_f3, d_gamma);
    halfStep(n_grids, d_u1, d_u2, d_u3, d_u1_temp, d_u2_temp, d_u3_temp, d_f1, d_f2, d_f3, d_tau, d_h);
    deviceBoundaryCondition(n_grids, d_u1_temp, d_u2_temp, d_u3_temp);
    updateFlux(n_grids, d_u1_temp, d_u2_temp, d_u3_temp, d_f1, d_f2, d_f3, d_gamma);
    step(n_grids, d_u1, d_u2, d_u3, d_u1_temp, d_u2_temp, d_u3_temp, d_f1, d_f2, d_f3, d_tau, d_h);
    updateU(n_grids, d_u1, d_u2, d_u3, d_u1_temp, d_u2_temp, d_u3_temp);
    deviceBoundaryCondition(n_grids, d_u1, d_u2, d_u3);
}
