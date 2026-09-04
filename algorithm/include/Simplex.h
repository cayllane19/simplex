#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/UmfPackSupport>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "constants.hpp"

using namespace Eigen;
using namespace std;

struct ETA
{
    VectorXd col;
    int idx;
};

struct EnteringVariable
{
    int idx;
    double reducedCost;
};

struct LeavingVariable
{
    int idx;
    double step;
};

class Simplex
{
public:
    Simplex(MatrixXd A, VectorXd b, VectorXd c, VectorXd lb, VectorXd ub, int numRows, int numCols);

    // Resolução do simplex
    int solvePhase0();
    double solve();

    // Informações da instância
    int numRows, numCols;
    MatrixXd A;
    VectorXd b;
    VectorXd c;
    VectorXd lb;
    VectorXd ub;

    // Definição das bases
    MatrixXd B;
    MatrixXd N;
    VectorXd cB;
    VectorXd cNB;

    // Solução
    VectorXd x;
    VectorXd xBasic;
    VectorXd xNonBasic;
    vector<int> basicVariables;
    vector<int> nonBasicVariables;

    // Fatoração ETA da base
    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> B0_lu;
    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> B0_trans_lu;
    vector<ETA> etaVector;

    // Funções core do simplex
    EnteringVariable ChooseVariableEnteringBasis();
    VectorXd CalculateDirection(EnteringVariable VariableEnteringBasis);
    LeavingVariable ChooseVariableLeavingBasis(VectorXd d, EnteringVariable VariableEnteringBasis);
    void UpdateSolution(double step, VectorXd d, EnteringVariable VariableEnteringBasis);
    void UpdateBasis(int idxVariableEnteringBasis, int idxVariableLeavingBasis);

    // BTRAN e FTRAN
    VectorXd BTRAN(VectorXd cB_input);
    VectorXd FTRAN(VectorXd a_input);
};