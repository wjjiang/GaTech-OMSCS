#include <algorithm>	// used for std::for_each, std::transform
#include <array>		// used for saving constant-size vectors
#include <functional>	// used for std::plus
#include <iostream>
#include <numeric>		// used for std::accumulate, std::inner_product
#include <random>		// used for generating -1/1 random numbers
#include <vector>		// used for saving MDP sequence, since its length varies
using namespace std;

// Observation 2D array
array<double, 5> X_B = { 1., 0., 0., 0., 0. };
array<double, 5> X_C = { 0., 1., 0., 0., 0. };
array<double, 5> X_D = { 0., 0., 1., 0., 0. };
array<double, 5> X_E = { 0., 0., 0., 1., 0. };
array<double, 5> X_F = { 0., 0., 0., 0., 1. };
array<array<double, 5>, 5> X_Matrix = {X_B, X_C, X_D, X_E, X_F};


// Function 1: calculate RMSE
// Step 1: lambda function to calculate the squared error between 2 elements
auto squaredError = [](auto weight, auto theo_weight) {
	auto error = weight - theo_weight;
	return (error * error);
};
// Step 2: calcualte RMSE using inner_product with 2 overheads:
// overhead 1: plus; overhead 2: squaredError lambda function
// logic: value = op1(value, op2(*first1, *first2))
double rmse_calc(array<double, 5>& Weights, array<double, 5>& Theo_Weights) {
	// calculate squared sum of errors
	// transform_reduce() is valid only at C++17
	// inner_product() will do the same trick
	double sum = inner_product(Weights.begin(), Weights.end(), Theo_Weights.begin(), 0.0, 
							   plus<>(), squaredError);
	// calculate root of sum
	double rmse = sqrt(sum / 5);

	return rmse;
}

// Function 2: generate MDP path from state D to either A or G
vector<int> MDP() {
	// Start point: x_D, which has index of 2 in [x_B, x_C, x_D, x_E, x_F] matrix
	int index = 2;
	// Record into MDP sequence
	vector<int> MDP_sequence = { index };

	// Setup random number generator
	random_device rd;
	mt19937_64 gen(rd());
	uniform_int_distribution<> dis(0, 1);

	// Update position based on random walk
	while ((index >= 0) && (index <= 4)) {
		// Random number -1 or 1
		int index_move = 2 * dis(gen) - 1;
		// Move index
		index += index_move;
		// Record into MDP sequence
		MDP_sequence.emplace_back(index);
	}

	// Return
	return MDP_sequence;
}

// Function 3: generate 100 training sets, each set has 10 sequences
array<array<vector<int>, 10>, 100> MDP_Training() {
	array<array<vector<int>, 10>, 100> Training_Sets;
	for (size_t i = 0; i<Training_Sets.size(); i++) {
		for (size_t j = 0; j<Training_Sets[i].size(); j++) {
			Training_Sets[i][j] = MDP();
		}
	}
	return Training_Sets;
}

// Function 4: prediction function P_t=np.dot(w', x_t)
// notice x_t here is index notation
// special treatment for x_A (z=0) and x_G (z=1)
double Prediction(array<double, 5>& Weights, int X_t) {
	// special case 1: x_A -> z=0
	if (X_t == -1) {
		return 0.0;
	}
	// special case 2: x_G -> z=1
	else if (X_t == 5) {
		return 1.0;
	}
	// otherwise P_t=np.dot(w', x_t)
	else {
		return Weights[X_t];
	}
}

// Function 5:calculation of delta_w at step t as shown in Eq. (4) of page 15
array<double, 5> Delta_w_t(double alpha, double p_t, double p_tPlus1, array<double, 5>& Summation) {
	array<double, 5> new_Summation;
	for (size_t i = 0; i < 5; i++) {
		new_Summation[i] = alpha * (p_tPlus1 - p_t) * Summation[i];
	}
	return new_Summation;
}

// Function 6: calculation of the weight incrementation for each MDP_sequence
array<double, 5> Delta_Weight_Seq(double alpha, double lambda, array<double, 5>& Weights,
								  vector<int>& MDP_sequence) {
	// Step 1: initialization
	// Step 1.1: initialize Delta_w array
	array<double, 5> Delta_w = {0.0, 0.0, 0.0, 0.0, 0.0};
	// Step 1.2: update observation, in dex notation, then calculate P_t
	int observation = MDP_sequence[0];
	double p_t = Prediction(Weights, observation);
	// Step 1.3: initialize Summation as the 1st element of MDP_sequence
	// notice elements in MDP_sequence corresponds to x_B ~ x_F
	// based on their indices in x_matrix
	array<double, 5> Summation = X_Matrix[MDP_sequence[0]];
	// Step 2: loop all element in MDP_sequence
	for (size_t i = 1; i < MDP_sequence.size(); i++) {
		// Step 2.1: update observation, in index notation
		observation = MDP_sequence[i];
		// Step 2.2: calculate p_(t+1)
		double p_tPlus1 = Prediction(Weights, observation);
		// Step 2.3: accumulate Delta_w
		auto Delta_wt = Delta_w_t(alpha, p_t, p_tPlus1, Summation);
		transform(Delta_w.begin(), Delta_w.end(), Delta_wt.begin(), Delta_w.begin(), plus<double>());
		// Step 2.4: update p_t wtih p_(t+1)
		p_t = p_tPlus1;
		// Step 2.5: update Summation except the last one
		// because X_matrix[X_A] and X_matrix[X_G] don't exist
		if (i != MDP_sequence.size()-1) {
			for (size_t j = 0; j < 5; j++) {
				Summation[j] = X_Matrix[observation][j] + lambda * Summation[j];
			}
		}
	}
	// Step 3: return
	return Delta_w;
}


///////////////////////////////////////////////////////////////////////
//
// Used for Fig. 3
//
///////////////////////////////////////////////////////////////////////

// Function 7: repeated presentation training paradigm
// As shown in page 20
// Used for generating Fig. 3
array<double, 5> Weight_repeated_presentation_training(double alpha, double lambda, double epsilon,
													   array<vector<int>, 10>& Training_Set) {
	// initialize Weights array
	array<double, 5> Weights = {0.5, 0.5, 0.5, 0.5, 0.5};
	// loop until converged
	while (true) {
		// initialize Delta_w array
		array<double, 5> Delta_w = {0.0, 0.0, 0.0, 0.0, 0.0};
		// accumulate Delta_w accross all 10 MDP_sequences in Training_Set
		for (vector<int> MDP_sequence : Training_Set) {
			auto Delta_wt = Delta_Weight_Seq(alpha, lambda, Weights, MDP_sequence);
			// accumulate
			transform(Delta_w.begin(), Delta_w.end(), Delta_wt.begin(), Delta_w.begin(), plus<double>());
		}
		// update Weights array
		transform(Weights.begin(), Weights.end(), Delta_w.begin(), Weights.begin(), plus<double>());
		// exit condition
		if (sqrt(inner_product(Delta_w.begin(), Delta_w.end(), Delta_w.begin(), 0.0)) < epsilon) {
			return Weights;
		}
	}
}

// Function 8: calculate the average of RMSE for the whole 100 Training_Sets
// Used for generating Fig. 3
double RMSE_Training_Sets_Fig3(double alpha, double lambda, double epsilon, 
							   array<double, 5>& Theo_Weights, 
							   array<array<vector<int>, 10>, 100>& Training_Sets) {
	// initialize RMSE_sum for average calculation
	double RMSE_sum = 0.0;
	// loop each of the 100 Training_Sets
	for (array<vector<int>, 10> Training_Set : Training_Sets) {
		// calculate Weight distribution
		array<double, 5> Weights = 
			Weight_repeated_presentation_training(alpha, lambda, epsilon, Training_Set);
		// calculate RMSE
		double RMSE = rmse_calc(Weights, Theo_Weights);
		// add to RMSE_sum
		RMSE_sum += RMSE;
	}
	// calculate average of RMSE
	double RMSE_avg = RMSE_sum / 100;
	// return
	return RMSE_avg;
}

// Function 9: calculate RMSE for each of the lambda used in Fig. 3
array<double, 11> RMSE_lambda_Fig3(double alpha, array<double, 11>& lambdas, double epsilon,
								   array<double, 5>& Theo_Weights,
								   array<array<vector<int>, 10>, 100>& Training_Sets) {
	// initialize RMSE vector for each lambda
	array<double, 11> RMSE_lambda;
	// loop each lambda
	for (size_t i = 0; i < 11; i++) {
		// calculate RMSE_avg for each lambda
		RMSE_lambda[i] = RMSE_Training_Sets_Fig3(alpha, lambdas[i], epsilon, Theo_Weights, Training_Sets);
	}
	// return
	return RMSE_lambda;
}


///////////////////////////////////////////////////////////////////////
//
// Used for Fig. 4
//
///////////////////////////////////////////////////////////////////////

// Function 10: one presentation training paradigm
// can choose to add control of all weights within [0.,1.]
// As shown in page 21-22
// Used for generating Fig. 4
// lambda function: Weights[Weights > 1.0] = 1.0 & Weights[Weights < 0.0] = 0.0
auto control_range = [](auto weight) {
	if (weight > 1.0) {
		weight = 1.0;
	}
	else if (weight < 0.0) {
		weight = 0.0;
	}
	return weight;
};
array<double, 5> Weight_one_presentation_training(double alpha, double lambda, bool control,
												  array<vector<int>, 10>& Training_Set) {
	// initialize Weights array
	array<double, 5> Weights = { 0.5, 0.5, 0.5, 0.5, 0.5 };
	// loop each of the 10 MDP_sequences in the training_set
	for (vector<int> MDP_sequence : Training_Set) {
		// calculate Delta_w
		array<double, 5> Delta_w = Delta_Weight_Seq(alpha, lambda, Weights, MDP_sequence);
		// add to Weight distribution: Weights += Delta_w
		transform(Weights.begin(), Weights.end(), Delta_w.begin(), Weights.begin(), plus<double>());
		// control: any weights should be within [0., 1.]
		if (control) {
			for_each(Weights.begin(), Weights.end(), control_range);
		}
	}
	// return
	return Weights;
}

// Function 11: calculate the average of rmse for the whole 100 training sets
// Used for generating Fig. 4
double RMSE_Training_Sets_Fig4(double alpha, double lambda, bool control,
							   array<double, 5>& Theo_Weights,
							   array<array<vector<int>, 10>, 100>& Training_Sets) {
	// initialize RMSE_sum for average calculation
	double RMSE_sum = 0.0;
	// loop each of the 100 Training_Sets
	for (array<vector<int>, 10> Training_Set : Training_Sets) {
		// calculate Weight distribution
		array<double, 5> Weights = Weight_one_presentation_training(alpha, lambda, control, Training_Set);
		// calculate RMSE
		double RMSE = rmse_calc(Weights, Theo_Weights);
		// add to RMSE_sum
		RMSE_sum += RMSE;
	}
	// calculate average of RMSE
	double RMSE_avg = RMSE_sum / 100;
	// return
	return RMSE_avg;
}

// Function 12: calculate RMSE for all alphas and lambdas used in Fig. 4
array<array<double, 13>, 4> RMSE_alpha_lambda_Fig4(array<double, 13>& alphas, array<double, 4>& lambdas,
												   bool control, array<double, 5>& Theo_Weights, 
												   array<array<vector<int>, 10>, 100>& Training_Sets) {
	// initialize RMSE list
	// 4 rows for 4 different lambdas [0., 0.3, 0.8, 1.]
	// 13 columns for 13 alphas at each lambda [0., 0.05, ..., 0.6]
	array<array<double, 13>, 4> RMSE_alpha_lambda;
	// loop each lambda
	for (size_t i = 0; i < 4; i++) {
		// loop each alpha
		for (size_t j = 0; j < 13; j++) {
			// fill into RMSE_alpha_lambda 2D array
			RMSE_alpha_lambda[i][j] = 
				RMSE_Training_Sets_Fig4(alphas[j], lambdas[i], control, Theo_Weights, Training_Sets);
		}
	}
	// return
	return RMSE_alpha_lambda;
}


///////////////////////////////////////////////////////////////////////
//
// Used for Fig. 5
//
///////////////////////////////////////////////////////////////////////

// Function 15: find best alpha based on specific lambda
// Used for generating Fig. 5
array<double, 11> RMSE_Best_alpha(array<double, 101>& alphas, array<double, 11>& lambdas,
								   bool control, array<double, 5>& Theo_Weights, 
								   array<array<vector<int>, 10>, 100>& Training_Sets) {
	// initialize RMSE_lambda
	array<double, 11> RMSE_lambda;
	// initialize RMSE_alpha for each lambda
	array<double, 101> RMSE_alpha;
	// loop each lambda
	for (size_t i = 0; i < 11; i++) {
		// loop each alpha
		for (size_t j = 0; j < 101; j++) {
			RMSE_alpha[j] = 
				RMSE_Training_Sets_Fig4(alphas[j], lambdas[i], control, Theo_Weights, Training_Sets);
		}
		// record the smallest RMSE_alpha into RMSE_lambda
		auto smallest = min_element(RMSE_alpha.begin(), RMSE_alpha.end());
		RMSE_lambda[i] = *smallest;
	}
	// return
	return RMSE_lambda;
}



int main() {
	array<array<vector<int>, 10>, 100> Training_Sets = MDP_Training();
	double alpha = 0.01;
	double lambda = 0.8;
	double epsilon = 0.001;
	array<double, 5> Theo_Weights = { 1./6., 1./3., 1./2., 2./3., 5./6. };

	// Fig. 3
	array<double, 11> lambdas_Fig3 = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
	array<double, 11> RMSE_lambda = 
		RMSE_lambda_Fig3(alpha, lambdas_Fig3, epsilon, Theo_Weights, Training_Sets);
	cout << "RMSE vs. Lambdas for Fig. 3: \n(";
	for_each(RMSE_lambda.begin(), RMSE_lambda.end(), [](double ele) {cout << ele << ", ";});
	cout << ")\n";
	
	// Fig. 4
	array<double, 4> lambdas_Fig4 = { 0., 0.3, 0.8, 1. };
	array<double, 13> alphas_Fig4;
	// generate alphas from 0.0 to 0.6, separated by 0.05
	iota(alphas_Fig4.begin(), alphas_Fig4.end(), 0.0);
	transform(alphas_Fig4.begin(), alphas_Fig4.end(), alphas_Fig4.begin(), 
			  [](double ele) {return ele*0.05;});
	array<array<double, 13>, 4> RMSE_alpha_lambda = 
		RMSE_alpha_lambda_Fig4(alphas_Fig4, lambdas_Fig4, true, Theo_Weights, Training_Sets);
	cout << "\nRMSE vs. Alphas for different Lambdas, for Fig. 4:\n";
	for (array<double, 13> RMSE_alpha : RMSE_alpha_lambda) {
		cout << "(";
		for_each(RMSE_alpha.begin(), RMSE_alpha.end(), [](double ele) {cout << ele << ", ";});
		cout << ")\n";
	}

	// Fig. 5
	array<double, 101> alphas_Fig5;
	// generate alphas from 0.00 to 1.00, separated by 0.01
	iota(alphas_Fig5.begin(), alphas_Fig5.end(), 0.0);
	transform(alphas_Fig5.begin(), alphas_Fig5.end(), alphas_Fig5.begin(),
		[](double ele) {return ele*0.01;});
	array<double, 11> lambdas_Fig5;
	// generate lambdas from 0.0 to 1.0, separated by 0.1
	iota(lambdas_Fig5.begin(), lambdas_Fig5.end(), 0.0);
	transform(lambdas_Fig5.begin(), lambdas_Fig5.end(), lambdas_Fig5.begin(),
		[](double ele) {return ele*0.1; });
	array<double, 11> RMSE_alpha = 
		RMSE_Best_alpha(alphas_Fig5, lambdas_Fig5, true, Theo_Weights, Training_Sets);
	cout << "\nRMSE vs. best Alphas, for Fig. 5: \n(";
	for_each(RMSE_alpha.begin(), RMSE_alpha.end(), [](double ele) {cout << ele << ", "; });
	cout << ")\n";


	system("pause");
	return 0;
}