import numpy as np
import time
#import random

#search part and cross validation part is separate 
# k-fold cross validation 
# accuracy = number of correct classifications/number of instances in our database 
# K = number of rows 
#first column: class label will be 1 or 2
#second column up to last are the features (pos or neg), each column represents a feature
def leave_one_out_cross_validation(orig_data, current_set, feature_to_add):
    #what i need to add to this is to edit the data file to delete the features not considered
    # features not within the current set should be ignored 
    # set the features not within the current set all to 0 to ignore them (columns)
    # ex, current set is 1,4,7 and there are 10 columns of features --> make rows 2,3,5... temporarily 0
    selected_features = current_set
    if feature_to_add is not None:
        selected_features = current_set + [feature_to_add]  
  # No addition in backward elimination
    data = np.zeros_like(orig_data)
    data[:, 0] = orig_data[:, 0]  # keep class labels
    data[:, selected_features] = orig_data[:, selected_features]  # keep only selected features

    number_correctly_classified = 0
    for i in range(1, data.shape[0]): #number of rows in data
        object_to_classify = data[i, 1:] #array of features
        #but it does not have to go through all of the features, only the ones in current set?
        label_object_to_classify = data[i, 0] #int in first column --> label (1 or 2)
        #print("Looping over i, at the ", i, "location")
        #print("The", i, "th object is in class ", label_object_to_classify)

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        for k in range(1, data.shape[0]):
            if k != i:
                # print(f"Ask if {i} is nearest neighbor with {k}")
                # vector object - feature vector row k = array of differences
                difference = object_to_classify - data[k, 1:]
                # **2 squares of each element in difference array
                squared_difference = difference ** 2
                # sum all of the pow2 differences 
                sum_of_squares = np.sum(squared_difference)
                # take sqrt of sum of squared differences --> euclidian distance
                distance = np.sqrt(sum_of_squares)
                #distance = np.linalg.norm(object_to_classify - data[k, 1:])
                #distance = np.sqrt(np.sum((object_to_classify - data[k, 1:]) ** 2)) 
                if distance < nearest_neighbor_distance: #smallest distance
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location, 0]
        # print("Object ", i, " is class ", label_object_to_classify)
        # print("its nearest neighbor is ", nearest_neighbor_location, " which is in class ", nearest_neighbor_label)

        if label_object_to_classify == nearest_neighbor_label: # if it is feature 2 and it is actually feature 2 then that is correct 
            number_correctly_classified += 1
    accuracy = number_correctly_classified / data.shape[0] #number i got correct / number i could have got correct
    return accuracy

#search algorithm
def feature_search_demo(data): #data is the dataset 
    print("This dataset has",data.shape[1] - 1,"features with", data.shape[0],"instances")
    print("Beginning FORWARD SELETION search")

    #calculate default rate first 
    class_labels = data[:, 0]
    count_ones = np.sum(class_labels == 1.0)
    count_twos = np.sum(class_labels == 2.0)
    default_rate = max(count_ones,count_twos)/data.shape[0]
    print(f"Default Rate [] is {default_rate*100:.1f}%")

    best_feature_set = []
    best_accuracy = 0
    current_set_of_features = [] #start with empty set of no featues (forward selection)
    for i in range(1, data.shape[1]): #number of columns - 1 --> should iterate through set of features 
        print(f"On the {i}th level of the search tree") #walk down the level of the tree
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0

        for k in range(1, data.shape[1]):  #number of column - 1
            if k not in current_set_of_features: #only consider adding if not already added
                print(f"--Considering adding the {k} feature") #at each level look at the remaining features
                # accuracy = random.uniform(0, 1) #random for now 
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k) #accuracy = cross validation accuracy
                print(f"Using feature(s) {current_set_of_features + [k]} accuracy is {accuracy*100:.1f}%")
                
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k #remember the feature (k) that gives us that accuracy
                    #so feature_to_add_at_this_level only 1 feature??

        current_set_of_features.append(feature_to_add_at_this_level)
        print(f"On level {i} i added feature {feature_to_add_at_this_level} to current set, {current_set_of_features} with accuracy {best_so_far_accuracy*100:.1f}%") #display the highest random number --> best feature to print
        
        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_feature_set = current_set_of_features.copy()  # Store a copy of the best feature set
    print("Finished search")
    print(f"Feature set {best_feature_set} was best, accuracy is {best_accuracy * 100:.1f}%") 
    return 

def backward_elimination(data):
    print("This dataset has",data.shape[1] - 1,"features with", data.shape[0],"instances")
    print("Beginning BACKWARD ELIMINATION search")

    current_set_of_features = list(range(1, data.shape[1]))
    #print(current_set_of_features)
    best_feature_set = current_set_of_features.copy()
    best_accuracy = leave_one_out_cross_validation(data, current_set_of_features, feature_to_add=None)
    print(f"On the {0}th level of the search tree") #walk down the level of the tree
    print(f"Using feature(s) {best_feature_set} accuracy is {best_accuracy*100:.1f}%")
              
    for i in range(1, data.shape[1]): #number of columns - 1 --> should iterate through set of features 
        print(f"On the {i}th level of the search tree") #walk down the level of the tree
        feature_to_remove_at_this_level = 0
        best_so_far_accuracy = 0

        for k in current_set_of_features:  #number of column - 1
            print(f"--Considering removing the {k} feature") #at each level look at the remaining features
            # accuracy = random.uniform(0, 1) #random for now 
            temp_set = current_set_of_features.copy()
            temp_set.remove(k)
            accuracy = leave_one_out_cross_validation(data, temp_set, None) #accuracy = cross validation accuracy
            print(f"Using feature(s) {temp_set} accuracy is {accuracy*100:.1f}%")
                
            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_remove_at_this_level = k #remember the feature removed (k) that gives us that accuracy

        current_set_of_features.remove(feature_to_remove_at_this_level)
        print(f"On level {i} i removed {feature_to_remove_at_this_level} for current set {current_set_of_features}, accuracy is {best_so_far_accuracy*100:.1f}%") #display the highest random number --> best feature to print
        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_feature_set = current_set_of_features.copy()  # Store a copy of the best feature set

    print("Finished search")
    print(f"Feature set {best_feature_set} was best, accuracy is {best_accuracy * 100:.1f}%") 
    return

def main():
    print("Welcome to Ramya's Feature Selection Algorithm")
    # name = input("Type in the name of the file to test: ")
    # data = np.loadtxt(name)
    data = np.loadtxt('CS170_Large_Data__98.txt')
    print("\n(1) Forward Selection \n(2) Backward Elimination")
    choice = input("Which algorithm would you like to run?  ")

    start_time = time.time()
    if choice == '1':
        feature_search_demo(data)
    if choice == '2':
        backward_elimination(data)
    end_time = time.time()
    elapsed_time = end_time - start_time  # calculate elapsed time
    print(f"\nTime: {elapsed_time:.2f} seconds")  

    return 

main()

#make code faster 
#vectorization 
# write with no loops impleticatly loop 
# caching - those numbers - few seconds 
