#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <ctime>
//#include <bits/stdc++.h>

using namespace std;

// Leave-One-Out Cross Validation Function
double leave_one_out_cross_validation(const vector<vector<double> >& orig_data, const vector<int>& current_set, int feature_to_add) {
    vector<vector<double> > data = orig_data;

    vector<int> selected_features = current_set;
    if (feature_to_add != -1) {
        selected_features.push_back(feature_to_add);
    }

    for (int i = 0; i < data.size(); ++i) {
        for (int j = 1; j < data[i].size(); ++j) {
            if (find(selected_features.begin(), selected_features.end(), j) == selected_features.end()) {
                data[i][j] = 0; // Ignore unselected features
            }
        }
    }

    int number_correctly_classified = 0;

    for (int i = 1; i < data.size(); ++i) {
        vector<double> object_to_classify = data[i];
        double label_object_to_classify = data[i][0];

        double nearest_neighbor_distance = numeric_limits<double>::infinity();
        int nearest_neighbor_location = -1;
        double nearest_neighbor_label = -1;

        for (int k = 0; k < data.size(); ++k) {
            if (k != i) {
                double distance = 0.0;
                for (int j = 1; j < data[i].size(); ++j) {
                    distance += pow(object_to_classify[j] - data[k][j], 2);
                }
                distance = sqrt(distance);

                if (distance < nearest_neighbor_distance) {
                    nearest_neighbor_distance = distance;
                    nearest_neighbor_location = k;
                    nearest_neighbor_label = data[k][0];
                }
            }
        }

        if (label_object_to_classify == nearest_neighbor_label) {
            number_correctly_classified++;
        }
    }

    return static_cast<double>(number_correctly_classified) / data.size();
}

// Forward Selection Algorithm
void feature_search_demo(const vector<vector<double> >& data) {
    cout << "This dataset has " << data[0].size() - 1 << " features with " << data.size() << " instances" << endl;
    cout << "Beginning FORWARD SELECTION search" << endl;

    int count_ones = 0, count_twos = 0;
    for (const auto& row : data) {
        if (row[0] == 1.0) count_ones++;
        else count_twos++;
    }

    double default_rate = max(count_ones, count_twos) / static_cast<double>(data.size());
    cout << "Default Rate [] is " << default_rate * 100 << "%" << endl;

    vector<int> best_feature_set;
    double best_accuracy = 0.0;

    vector<int> current_set_of_features;

    for (int i = 1; i < data[0].size(); ++i) {
        cout << "On the " << i << "th level of the search tree" << endl;

        int feature_to_add_at_this_level = -1;
        double best_so_far_accuracy = 0.0;

        for (int k = 1; k < data[0].size(); ++k) {
            if (find(current_set_of_features.begin(), current_set_of_features.end(), k) == current_set_of_features.end()) {
                cout << "--Considering adding the " << k << " feature" << endl;

                double accuracy = leave_one_out_cross_validation(data, current_set_of_features, k);
                cout << "Using feature(s) ";
                for (int feat : current_set_of_features) cout << feat << " ";
                cout << k << " accuracy is " << accuracy * 100 << "%" << endl;

                if (accuracy > best_so_far_accuracy) {
                    best_so_far_accuracy = accuracy;
                    feature_to_add_at_this_level = k;
                }
            }
        }

        if (feature_to_add_at_this_level != -1) {
            current_set_of_features.push_back(feature_to_add_at_this_level);
            cout << "On level " << i << " i added feature " << feature_to_add_at_this_level << " to current set, accuracy is " << best_so_far_accuracy * 100 << "%" << endl;

            if (best_so_far_accuracy > best_accuracy) {
                best_accuracy = best_so_far_accuracy;
                best_feature_set = current_set_of_features;
            }
        }
    }

    cout << "Finished search" << endl;
    cout << "Feature set: ";
    for (int feat : best_feature_set) cout << feat << " ";
    cout << "was best, accuracy is " << best_accuracy * 100 << "%" << endl;
}

// Backward Elimination Algorithm
void backward_elimination(const vector<vector<double> >& data) {
    cout << "Beginning BACKWARD ELIMINATION search" << endl;

    vector<int> current_set_of_features;
    for (int i = 1; i < data[0].size(); ++i) {
        current_set_of_features.push_back(i);
    }

    vector<int> best_feature_set = current_set_of_features;
    double best_accuracy = leave_one_out_cross_validation(data, current_set_of_features, -1);

    for (int i = 1; i < data[0].size(); ++i) {
        cout << "On the " << i << "th level of the search tree" << endl;

        int feature_to_remove_at_this_level = -1;
        double best_so_far_accuracy = 0.0;

        for (int k : current_set_of_features) {
            vector<int> temp_set = current_set_of_features;
            temp_set.erase(remove(temp_set.begin(), temp_set.end(), k), temp_set.end());

            double accuracy = leave_one_out_cross_validation(data, temp_set, -1);
            cout << "Using feature(s) ";
            for (int feat : temp_set) cout << feat << " ";
            cout << "accuracy is " << accuracy * 100 << "%" << endl;

            if (accuracy > best_so_far_accuracy) {
                best_so_far_accuracy = accuracy;
                feature_to_remove_at_this_level = k;
            }
        }

        current_set_of_features.erase(remove(current_set_of_features.begin(), current_set_of_features.end(), feature_to_remove_at_this_level), current_set_of_features.end());

        if (best_so_far_accuracy > best_accuracy) {
            best_accuracy = best_so_far_accuracy;
            best_feature_set = current_set_of_features;
        }
    }

    cout << "Finished search" << endl;
    cout << "Feature set: ";
    for (int feat : best_feature_set) cout << feat << " ";
    cout << "was best, accuracy is " << best_accuracy * 100 << "%" << endl;
}

// Main function
int main() {
    cout << "Welcome to Ramya's Feature Selection Algorithm" << endl;

    vector<vector<double> > data;
    ifstream file("CS170_Large_Data__10.txt");

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    cout << "\n(1) Forward Selection" << endl;
    cout << "(2) Backward Elimination" << endl;
    cout << "Which algorithm would you like to run? ";

    int choice;
    cin >> choice;

    clock_t start_time = clock();

    if (choice == 1) {
        feature_search_demo(data);
    } else if (choice == 2) {
        backward_elimination(data);
    }

    clock_t end_time = clock();
    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "\nTime: " << elapsed_time << " seconds" << endl;

    return 0;
}
