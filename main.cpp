#include <iostream>

#include "Dataset.h"
#include "RandomForest.h"

using namespace std;

int main() {
	Dataset helen;
	helen.readFromFile("config/helen.cfg");

	for (int feature = 0; feature < 68; ++feature) {
		cout << "============================== Feature " << feature << "==============================" << endl;
		for (int stage = 0; stage < 6; ++stage) {
			cout << "------------------ Stage" << stage << "------------------" << endl;
			stringstream sstream;
			sstream << "model/random_forest/feature_" << feature << "_stage_" << stage << ".mdl";
			RandomForest forest;
			forest.generateFromDataset(helen, "config/train.cfg", stage, feature);
			forest.outputToFile(sstream.str());
		}
		cout << endl;
	}

	return 0;
}
