// main.cpp
#include <chrono>
#include <fmt/core.h>

#include "Simplex.h"
#include "mps_reader.hpp"
#include "params.hpp"
#include "preprocess.hpp"

using fmt::println;

int main(int argc, char* argv[]) {
    Params::parse(argc, argv);
    auto& p = Params::get();

    MpsReader reader;
    ProblemData problem_data = reader.read(p.instance_file);
    println("Problem data: {}", problem_data);

    // Passo 2: Preprocessamento
    Preprocessor preprocessor(p, problem_data);
    ProblemData preprocessed = preprocessor.process();
    println("Preprocessed data: {}", preprocessed);

    MatrixXd A = preprocessed.A;
    VectorXd b = preprocessed.b;
    VectorXd c = preprocessed.c;
    VectorXd lb = preprocessed.lb;
    VectorXd ub = preprocessed.ub;

    int numRows = preprocessed.m;
    int numCols = preprocessed.n + preprocessed.m;

    Simplex simplex(A, b, -c, lb, ub, numRows, numCols);

    int phase0 = simplex.solvePhase0();
    if (phase0 == 0) {
        auto cost = simplex.solve();
    } else {
        println("Problem is infeasible.");
    }

    return 0;
}