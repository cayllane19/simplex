#include "Simplex.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

Simplex::Simplex(MatrixXd A, VectorXd b, VectorXd c, VectorXd lb, VectorXd ub, int numRows, int numCols) {
    this->A = A;
    this->b = b;
    this->c = c;
    this->lb = lb;
    this->ub = ub;
    this->numRows = numRows;
    this->numCols = numCols;

    this->x = VectorXd::Zero(numCols);

    // Ajusta os bounds das variáveis adicionadas na base inicial
    if (this->lb.size() < numCols) {
        this->lb.conservativeResize(numCols); // ajusta sem apagar as informações
        this->ub.conservativeResize(numCols);

        int numOriginalVars = numCols - numRows;

        for (int j = numOriginalVars; j < numCols; ++j) {
            this->lb(j) = 0.0;
            this->ub(j) = pInf;
        }
    }
}

int Simplex::solvePhase0() {
    // Fase 0: encontrar uma solução factível e uma base inicial
    int numOriginalVars = numCols - numRows;

    this->basicVariables.clear();
    this->nonBasicVariables.clear();

    // As últimas numRows variáveis formam a base inicial
    for (int i = numOriginalVars; i < numCols; i++) {
        this->basicVariables.push_back(i);
    }

    for (int i = 0; i < numOriginalVars; i++) {
        this->nonBasicVariables.push_back(i);
    }

    // Inicializa as variáveis não básicas com o bound
    // Caso não exista, o valor inicial é zero.
    for (int i = 0; i < nonBasicVariables.size(); i++) {
        int j = nonBasicVariables[i];

        if (lb[j] > nInf) {
            x[j] = lb[j];

        } else if (ub[j] < pInf) {
            x[j] = ub[j];

        } else {
            x[j] = 0.0;
        }
    }

    xBasic.resize(basicVariables.size());
    xNonBasic.resize(nonBasicVariables.size());

    bool okPhase0 = false;
    int iter = 0;

    while (!okPhase0) {
        iter++;

        // Monta as matrizes B e N a partir das variáveis básicas e não básicas.
        this->B = MatrixXd(numRows, numRows);
        for (int i = 0; i < numRows; ++i) {
            B.col(i) = A.col(basicVariables[i]);
        }

        this->N = MatrixXd(numRows, numOriginalVars);
        for (int i = 0; i < numOriginalVars; ++i) {
            N.col(i) = A.col(nonBasicVariables[i]);
        }

        // Resolve B*xB = b - N*xN.
        for (int i = 0; i < nonBasicVariables.size(); ++i) {
            xNonBasic[i] = x[nonBasicVariables[i]];
        }

        VectorXd rightSide = b - N * xNonBasic;
        VectorXd x_B = B.fullPivLu().solve(rightSide);

        for (int i = 0; i < basicVariables.size(); i++) {
            x[basicVariables[i]] = x_B(i);
            xBasic[i] = x_B(i);
        }

        // P: variáveis abaixo do limite inferior.
        // Q: variáveis acima do limite superior.
        vector<int> P, Q;
        for (int i = 0; i < basicVariables.size(); i++) {
            int var = basicVariables[i];

            if (x[var] < lb[var] - EPSILON_1) {
                P.push_back(var);
            } else if (x[var] > ub[var] + EPSILON_1) {
                Q.push_back(var);
            }
        }

        // Se P e Q estiverem vazios, a solução é factível para a fase 0
        if (P.empty() && Q.empty()) {
            cout << "Vectors P and Q are empty; B is defined!" << endl;
            okPhase0 = true;
            break;
        }

        this->cB = VectorXd::Zero(basicVariables.size());
        this->cNB = VectorXd::Zero(nonBasicVariables.size());

        for (int i = 0; i < basicVariables.size(); ++i) {
            int var = basicVariables[i];

            if (x[var] < lb[var] - EPSILON_1) {
                cB[i] = 1;

            } else if (x[var] > ub[var] + EPSILON_1) {
                cB[i] = -1;
            }
        }

        Eigen::SparseMatrix<double> B_sparse = B.sparseView();
        B_sparse.makeCompressed();
        B0_lu.compute(B_sparse);

        Eigen::SparseMatrix<double> B_trans = B.transpose().sparseView();
        B_trans.makeCompressed();
        B0_trans_lu.compute(B_trans);

        etaVector.clear();

        EnteringVariable entering = ChooseVariableEnteringBasis();

        if (entering.idx == -1) {
            cout << "Impossible problem! It is not possible to eliminate the invisibilities." << endl;
            return -1;
        }

        VectorXd d = CalculateDirection(entering);

        LeavingVariable leaving = ChooseVariableLeavingBasis(d, entering);

        if (std::isinf(leaving.step)) {
            cout << "Unlimited Problem" << endl;
            return -1;
        }

        UpdateSolution(leaving.step, d, entering);

        if (leaving.idx != -1) {
            UpdateBasis(entering.idx, leaving.idx);
        }

        if (iter > 1000) {
            cout << "Fase 0: Limite máximo de iterações atingido." << endl;
            return -1;
        }
    }

    return 0;
}

double Simplex::solve() {
    // Atualiza os custos das variáveis básicas e não básicas
    cB.resize(basicVariables.size());
    for (int i = 0; i < basicVariables.size(); ++i) {
        cB[i] = c[basicVariables[i]];
    }

    cNB.resize(nonBasicVariables.size());
    for (int i = 0; i < nonBasicVariables.size(); ++i) {
        cNB[i] = c[nonBasicVariables[i]];
    }

    // Fatoração da matriz B e da matriz transposta
    Eigen::SparseMatrix<double> B_sparse = B.sparseView();
    B_sparse.makeCompressed();
    B0_lu.compute(B_sparse);

    Eigen::SparseMatrix<double> B_trans = B.transpose().sparseView();
    B_trans.makeCompressed();
    B0_trans_lu.compute(B_trans);

    etaVector.clear();

    VectorXd rightSide = b - N * xNonBasic;
    xBasic = B0_lu.solve(rightSide);

    bool isOptimal = false;
    int iterations = 0;

    while (!isOptimal) {
        iterations++;

        EnteringVariable VariableEnteringBasis = ChooseVariableEnteringBasis();

        if (VariableEnteringBasis.idx == -1) {
            cout << "Optimal found!" << endl;
            double z = cB.dot(xBasic);
            cout << "Z:" << z << endl;
            isOptimal = true;
            return z;
        }

        VectorXd d = CalculateDirection(VariableEnteringBasis);

        LeavingVariable VariableLeavingBasis = ChooseVariableLeavingBasis(d, VariableEnteringBasis);

        if (std::isinf(VariableLeavingBasis.step)) {
            cout << "Unlimited Problem" << endl;
            break;
        }

        UpdateSolution(VariableLeavingBasis.step, d, VariableEnteringBasis);

        if (VariableLeavingBasis.idx != -1) {
            etaVector.push_back(ETA{d, VariableLeavingBasis.idx});
            UpdateBasis(VariableEnteringBasis.idx, VariableLeavingBasis.idx);
        }
    }

    return 0;
}

EnteringVariable Simplex::ChooseVariableEnteringBasis() {
    // Calcula os multiplicadores duais: B^T*y = cB
    VectorXd dual = BTRAN(cB);

    // Custos reduzidos das variáveis não básicas
    VectorXd reducedCosts(nonBasicVariables.size());

    for (int i = 0; i < nonBasicVariables.size(); i++) {
        reducedCosts[i] = cNB[i] - dual.dot(N.col(i));
    }

    int idxVariableEnteringBasis = -1;
    double bestReducedCost = 0.0;

    for (int i = 0; i < reducedCosts.size(); i++) {
        int variable = nonBasicVariables[i];

        // Custo reduzido positivo: aumentar x_j pode melhorar a função objetivo
        if (reducedCosts[i] > EPSILON_1 && xNonBasic[i] < ub[variable]) {
            if (reducedCosts[i] > bestReducedCost) {
                bestReducedCost = reducedCosts[i];
                idxVariableEnteringBasis = i;
            }

            // Custo reduzido negativo: diminuir x_j pode melhorar a função objetivo
        } else if (reducedCosts[i] < -EPSILON_1 && xNonBasic[i] > lb[variable]) {
            if (reducedCosts[i] < bestReducedCost) {
                bestReducedCost = reducedCosts[i];
                idxVariableEnteringBasis = i;
            }
        }
    }

    return {idxVariableEnteringBasis, bestReducedCost};
}

VectorXd Simplex::CalculateDirection(EnteringVariable VariableEnteringBasis) {
    // Calculo da direção --> d = B^(-1) a_j
    VectorXd d = FTRAN(N.col(VariableEnteringBasis.idx));
    return d;
}

LeavingVariable Simplex::ChooseVariableLeavingBasis(VectorXd d, EnteringVariable VariableEnteringBasis) {
    int idxVariableLeavingBasis = -1;
    int variableNonBasic = nonBasicVariables[VariableEnteringBasis.idx];
    int smallerIdxStepBasic = -1;

    double enteringStep;
    double smallerBasicStep = pInf;
    double step;

    // Caso 1 - Custo reduzido positivo
    if (VariableEnteringBasis.reducedCost > EPSILON_1) {
        enteringStep = ub[variableNonBasic] - xNonBasic[VariableEnteringBasis.idx];

        // Análise das variáveis básicas
        for (int i = 0; i < d.size(); i++) {
            int variableBasic = basicVariables[i];
            double basicStep = pInf;

            if (d[i] > EPSILON_1) {
                basicStep = (xBasic[i] - lb[variableBasic]) / std::abs(d[i]);
            } else if (d[i] < -EPSILON_1) {
                basicStep = (ub[variableBasic] - xBasic[i]) / std::abs(d[i]);
            }

            if (basicStep < smallerBasicStep) {
                smallerIdxStepBasic = variableBasic;
                smallerBasicStep = basicStep;
                idxVariableLeavingBasis = i;
            } else if (basicStep == smallerBasicStep && variableBasic < smallerIdxStepBasic) {
                smallerIdxStepBasic = variableBasic;
                idxVariableLeavingBasis = i;
            }
        }

        if (smallerBasicStep < enteringStep) {
            step = smallerBasicStep;
        } else {
            step = enteringStep;
            idxVariableLeavingBasis = -1;
        }

        // Caso 2 - Custo reduzido negativo
    } else if (VariableEnteringBasis.reducedCost < -EPSILON_1) {
        enteringStep = xNonBasic[VariableEnteringBasis.idx] - lb[variableNonBasic];

        for (int i = 0; i < d.size(); i++) {
            int variableBasic = basicVariables[i];
            double basicStep = pInf;

            if (d[i] > EPSILON_1) {
                basicStep = (ub[variableBasic] - xBasic[i]) / std::abs(d[i]);
            } else if (d[i] < -EPSILON_1) {
                basicStep = (xBasic[i] - lb[variableBasic]) / std::abs(d[i]);
            }

            if (basicStep < smallerBasicStep) {
                smallerIdxStepBasic = variableBasic;
                smallerBasicStep = basicStep;
                idxVariableLeavingBasis = i;
            } else if (basicStep == smallerBasicStep && variableBasic < smallerIdxStepBasic) {
                smallerIdxStepBasic = variableBasic;
                idxVariableLeavingBasis = i;
            }
        }

        if (smallerBasicStep < enteringStep) {
            step = smallerBasicStep;
        } else {
            step = enteringStep; // não entra na base
            idxVariableLeavingBasis = -1;
        }
    }

    return {idxVariableLeavingBasis, step};
}

void Simplex::UpdateSolution(double step, VectorXd d, EnteringVariable VariableEnteringBasis) {
    if (VariableEnteringBasis.reducedCost > EPSILON_1) {
        xNonBasic[VariableEnteringBasis.idx] = xNonBasic[VariableEnteringBasis.idx] + step;
    } else {
        xNonBasic[VariableEnteringBasis.idx] = xNonBasic[VariableEnteringBasis.idx] - step;
    }

    x[nonBasicVariables[VariableEnteringBasis.idx]] = xNonBasic[VariableEnteringBasis.idx];

    for (int i = 0; i < d.size(); i++) {
        if (VariableEnteringBasis.reducedCost > EPSILON_1 && d[i] > EPSILON_1) {
            xBasic[i] = xBasic[i] - step * std::abs(d[i]);
        } else if (VariableEnteringBasis.reducedCost > EPSILON_1 && d[i] < -EPSILON_1) {
            xBasic[i] = xBasic[i] + step * std::abs(d[i]);
        } else if (VariableEnteringBasis.reducedCost < -EPSILON_1 && d[i] > EPSILON_1) {
            xBasic[i] = xBasic[i] + step * std::abs(d[i]);
        } else if (VariableEnteringBasis.reducedCost < -EPSILON_1 && d[i] < -EPSILON_1) {
            xBasic[i] = xBasic[i] - step * std::abs(d[i]);
        }
        x[basicVariables[i]] = xBasic[i];
    }
}

void Simplex::UpdateBasis(int idxVariableEnteringBasis, int idxVariableLeavingBasis) {
    int entering = nonBasicVariables[idxVariableEnteringBasis];
    int leaving = basicVariables[idxVariableLeavingBasis];

    basicVariables[idxVariableLeavingBasis] = entering;
    nonBasicVariables[idxVariableEnteringBasis] = leaving;

    cB[idxVariableLeavingBasis] = c[entering];
    cNB[idxVariableEnteringBasis] = c[leaving];

    double enteringValue = xNonBasic[idxVariableEnteringBasis];
    xNonBasic[idxVariableEnteringBasis] = xBasic[idxVariableLeavingBasis];
    xBasic[idxVariableLeavingBasis] = enteringValue;

    N.col(idxVariableEnteringBasis) = A.col(leaving);
    B.col(idxVariableLeavingBasis) = A.col(entering);
}

VectorXd Simplex::FTRAN(VectorXd a_input) {
    // Resolve inicialmente usando a fatoração da base inicial
    VectorXd v = B0_lu.solve(a_input);

    // Aplica as transformações ETA na ordem em que foram realizadas
    for (const ETA& e : etaVector) {
        VectorXd x(v.size());
        int r = e.idx;

        x[r] = v[r] / e.col[r];

        for (int j = 0; j < v.size(); j++) {
            if (j != r) {
                x[j] = v[j] - x[r] * e.col[j];
            }
        }
        v = x;
    }
    return v;
}

VectorXd Simplex::BTRAN(VectorXd cB_input) {
    VectorXd w = cB_input;

    // Aplica as transformações ETA da mais recente para a mais antiga
    for (auto iter = etaVector.rbegin(); iter != etaVector.rend(); ++iter) {
        VectorXd x(w.size());
        int r = iter->idx;

        for (int j = 0; j < w.size(); j++) {
            if (j != r) {
                x[j] = w[j];
            }
        }

        double sum = 0.0;
        for (int j = 0; j < w.size(); j++) {
            if (j != r) {
                sum += x[j] * iter->col[j];
            }
        }
        x[r] = (w[r] - sum) / iter->col[r];
        w = x;
    }
    return B0_trans_lu.solve(w);
}