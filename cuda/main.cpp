#include "shock_tube.h"
#include <iostream>
#include <string>
#include <cstdlib>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "CUDA implementation of Lax-Wendroff scheme to solve Sod's 1D Shock Tube Problem\n\n";
    std::cout << "Options:\n";
    std::cout << "  -n <grids>     Number of grid points (default: 200)\n";
    std::cout << "  -t <time>      Final simulation time (default: 0.2)\n";
    std::cout << "  -o <prefix>    Output file prefix (default: solution)\n";
    std::cout << "  -steps <n>     Number of output time steps (default: 21)\n";
    std::cout << "  --host         Run on host (CPU) only\n";
    std::cout << "  --device       Run on device (GPU) only\n";
    std::cout << "  --test         Run unit tests\n";
    std::cout << "  --help         Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Default parameters
    int n_grids = 200;
    double t_max = 0.2;
    std::string output_prefix = "solution";
    int n_output = 21;
    bool run_host = false;
    bool run_device = false;
    bool run_tests = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--test") {
            run_tests = true;
        } else if (arg == "--host") {
            run_host = true;
        } else if (arg == "--device") {
            run_device = true;
        } else if (arg == "-n" && i + 1 < argc) {
            n_grids = std::atoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            t_max = std::atof(argv[++i]);
        } else if (arg == "-o" && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (arg == "-steps" && i + 1 < argc) {
            n_output = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Validate grid size
    if (n_grids < 10) {
        std::cerr << "Error: Grid size must be at least 10." << std::endl;
        return 1;
    }
    
    try {
        // Create shock tube simulation
        ShockTube simulation(n_grids);
        
        if (run_tests) {
            // Run unit tests
            simulation.runTests();
        } else if (run_host) {
            // Run host simulation only
            std::string filename = output_prefix + "_lax_host.dat";
            simulation.runHostSimulation(t_max, filename);
        } else if (run_device) {
            // Run device simulation only
            simulation.runDeviceSimulation(t_max, output_prefix + "_lax_device", n_output);
        } else {
            // Run both host and device simulations
            simulation.runSimulation(t_max, output_prefix, n_output);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
