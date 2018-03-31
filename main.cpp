#include <iostream>

#include "Dataset.h"
#include "RandomForest.h"
#include "Regressor.h"

using namespace std;

int main() {
	Params_ params("config/train.cfg");
	params.output();

	// Dataset helen;
	// helen.readFromFile("config/helen.cfg", false);

	Regressor regressor(params);
	// regressor.Train(helen);
	regressor.readModels();
	regressor.predictImage("E:/5.jpg");
	
	return 0;
}
