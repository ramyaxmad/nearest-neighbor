import numpy as np
import random
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
    selected_features = current_set + [feature_to_add]  # Include the new feature
    data = np.zeros_like(orig_data)
    data[:, 0] = orig_data[:, 0]  # Preserve class labels
    data[:, selected_features] = orig_data[:, selected_features]  # Keep only selected features

    
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

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / data.shape[0] #number i got correct / number i could have got correct
    return accuracy

#search algorithm
def feature_search_demo(data): #data is the dataset 
    print("This dataset has",data.shape[1] - 1,"features with", data.shape[0],"instances")
    print("Beginning search")
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
                print(f"Using feature(s) {current_set_of_features + [k]} accuracy is {accuracy*100}%")
                
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k #remember the feature (k) that gives us that accuracy
                    #so feature_to_add_at_this_level only 1 feature??

        current_set_of_features.append(feature_to_add_at_this_level)
        print(f"On level {i} i added feature {feature_to_add_at_this_level} to current set") #display the highest random number --> best feature to print
        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_feature_set = current_set_of_features.copy()  # Store a copy of the best feature set

    print("Finished search")
    print(f"Feature set {best_feature_set} was best, accuracy is {best_accuracy * 100}%") 
    return 

def main():
    print("Welcome to Ramya's Feature Selection Algorithm")
    # name = input("Type in the name of the file to test: ")
    # data = np.loadtxt(name)

    print("\n(1) Forward Selection \n(2) Backward Elimination")
    choice = input("Which algorithm would you like to run?  ")
    data = np.loadtxt('CS170_Small_Data__90.txt')
    if choice == '1':
        feature_search_demo(data)
    if choice == '2':
        pass
    
    return 

main()