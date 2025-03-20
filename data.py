import numpy as np
import time 

def leave_one_out_cross_validation(orig_data, current_set, feature_to_add):
    row_num = len(orig_data)

    selected_features = current_set
    if feature_to_add is not None: #for forward selection
        selected_features = current_set + [feature_to_add]
    
    #for filtering 
    data = np.zeros_like(orig_data)
    for i in range(row_num):  # Loop over rows
        data[i, 0] = orig_data[i, 0] #keep first column
    for i in range(1, row_num): # copy the selected features   
        for j in selected_features:         
            data[i, j] = orig_data[i, j]

    number_correctly_classified = 0
    for i in range(1, row_num):
        object_to_classify = data[i, 1:]  #array of features
        label_object_to_classify = data[i, 0] #int in first column --> label (1 or 2)
    
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(1, row_num):
            if k != i: 
                distance = np.linalg.norm(object_to_classify - data[k, 1:]) #euclidean distance
                if distance < nearest_neighbor_distance: #smallest distance
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location, 0] #class of the nearest neighbor k
        
        if label_object_to_classify == nearest_neighbor_label: # if it is feature 2 and it is actually feature 2 then that is correct 
            number_correctly_classified += 1
    accuracy = number_correctly_classified / row_num #number i got correct / number i could have got correct
    return accuracy

def feature_search_demo(data): #forward selection
    row_num = len(data)
    col_num = len(data[0])

    #calculate default rate first 
    class_labels = data[:, 0]
    count_ones = np.sum(class_labels == 1.0)
    count_twos = np.sum(class_labels == 2.0)
    default_rate = max(count_ones,count_twos)/row_num
    print(f"Default rate, using feature(s) {{}}, accuracy is {default_rate*100:.1f}%")
    print("Beginning Search.")

    best_feature_set = []
    best_accuracy = 0
    current_set_of_features = [] #start with empty set
    warning_printed = False  # track if the warning has been printed

    for i in range(1, col_num): #go from 2nd column to last
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0

        for k in range(1, col_num):
            if k not in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k) 
                if current_set_of_features: 
                    print(f"\tUsing feature(s) {{{', '.join(str(feature) for feature in current_set_of_features)}, {k}}} accuracy is {accuracy*100:.1f}%")
                else:
                    print(f"\tUsing feature(s) {{{k}}} accuracy is {accuracy*100:.1f}%")

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k #remember the feature (k) that gives us that accuracy  
                    
         # check if accuracy decreased
        if best_so_far_accuracy < best_accuracy and not warning_printed:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            warning_printed = True  # the warning is printed only once

        current_set_of_features.append(feature_to_add_at_this_level)
        if len(current_set_of_features) != col_num - 1:
            print(f"Feature set {{{', '.join(str(feature) for feature in current_set_of_features)}}} was best, accuracy is {best_so_far_accuracy*100:.1f}%")

        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_feature_set = current_set_of_features.copy()

    print(f"Finished search!! The best feature subset is {{{', '.join(str(feature) for feature in best_feature_set)}}}, which has an accuracy of {best_accuracy * 100:.1f}%")
    return

def backward_elimination(data):
    row_num = len(data)
    col_num = len(data[0])

    #find accuracy of full set 
    current_set_of_features = list(range(1, col_num))
    best_feature_set = current_set_of_features.copy()
    best_accuracy = leave_one_out_cross_validation(data, current_set_of_features, feature_to_add=None)
    # print(f"\tUsing feature(s) {{{', '.join(str(feature) for feature in best_feature_set)}}} accuracy is {best_accuracy * 100:.1f}%")
    
    print("Beginning Search.")
    for i in range(1, col_num):
        feature_to_remove_at_this_level = 0
        best_so_far_accuracy = 0

        for k in current_set_of_features:
            temp_set = current_set_of_features.copy()
            temp_set.remove(k) #in backwards you remove from the list
            accuracy = leave_one_out_cross_validation(data, temp_set, None) 
            print(f"\tUsing feature(s) {{{', '.join(str(feature) for feature in temp_set)}}} accuracy is {accuracy * 100:.1f}%")

            if accuracy > best_so_far_accuracy: #finds highest accuracy
                best_so_far_accuracy = accuracy
                feature_to_remove_at_this_level = k #remember the feature removed (k) that gives us that accuracy
        
        current_set_of_features.remove(feature_to_remove_at_this_level)
        if len(current_set_of_features) != 0:
            print(f"Feature set {{{', '.join(str(feature) for feature in current_set_of_features)}}} was best, accuracy is {best_so_far_accuracy*100:.1f}%")

        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_feature_set = current_set_of_features.copy()  # Store a copy of the best feature set
    
    print(f"Finished search!! The best feature subset is {{{', '.join(str(feature) for feature in best_feature_set)}}}, which has an accuracy of {best_accuracy * 100:.1f}%")
    return 

def main():
    print("Welcome to Ramya's Feature Selection Algorithm.")
    name = input("Type in the name of the file to test : ")
    data = np.loadtxt(name)
    #data = np.loadtxt('CS170_Small_Data__61.txt')

    print("\nType the number of the algorithm you want to run.")
    print("\t(1) Forward Selection \n\t(2) Backward Elimination")
    choice = input("Which algorithm would you like to run?  ")

    row_num = len(data)
    col_num = len(data[0])
    print("\nThis dataset has",col_num - 1,"features (not including the class attribute) with", row_num,"instances.")
    
    current_set = list(range(1, col_num))
    accuracy = leave_one_out_cross_validation(data, current_set, None)
    print(f"Running nearest neighbor with all {col_num - 1} features, using \"leave-one-out\" evaluation, I get an accuracy of {accuracy * 100:.1f}%")

    start_time = time.time()
    if choice == '1':
        feature_search_demo(data)
    if choice == '2':
        backward_elimination(data)
    end_time = time.time()
    elapsed_time = end_time - start_time  # calculate elapsed time
    print(f"Time: {elapsed_time:.1f} seconds")  

    return 

main()