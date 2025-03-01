#search part and cross validation part is divorced 

# k-fold cross validation 
#accuracy = number of correct classifications/number of instances in our database 
# K = number of rows 
import numpy as np

#data = some txt file --> loading an object into main memory 
data = np.loadtxt('CS170_Small_Data__99.txt')
def leave_one_out_cross_validation(data, current_set, feature_to_add):
    for i in range(data.shape[0]): #number of rows in data, iterates through each row
        object_to_classify = data[i, 1:] #array of features --> 2nd to the end features, not first in row
        label_object_to_classify = data[i, 0] #element in first column --> label
        print("Looping over i, at the ", i, "location")
        print("", i, "th object is in class ", label_object_to_classify)

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        for k in range(data.shape[0]):
            if k != i:
                print("Ask if ", i, " is nearest neighbor with ", k)
                # vector object - feature vector row k = array of differences
                # **2 squares of each element in difference array
                # sum all of the pow2 differences 
                # take sqrt of sum of squared differences --> euclidian distance
                distance = np.sqrt(np.sum((object_to_classify - data[k, 1:]) ** 2)) #calculate straight line space. take differences and 
                if distance < nearest_neighbor_distance: #best neighbor so far
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location, 0]
        print("Object ", i, " is class ", label_object_to_classify)
        print("its nearest neighbor is ", nearest_neighbor_location, " which is in class ", nearest_neighbor_label)
    return

#search algorithm - fix  
def feature_search_demo(data): #data is the dataset 
    current_set_of_features = []
    #first column of the dataset is class label and not a feature 
    for i in range(10): #number of columns - 1 
        print("on the ", i, "th level of the search tree") #walk down the level of the tree
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0

        for k in range(10):  #number of rows in column - 1 
            #once you have added the feature you should not add it again 
            #every time it goes through this loop its gonna have one thing to add 
            if k not in current_set_of_features: #only consider adding if not already added 
                print("--Considering adding the ", k, " feature") #at each level look at the remaining features
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k+1) #accuracy = cross validation accuracy

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_add_at_this_level = k #remember the feature (k) that gives us that accuracy
                #so feature_to_add_at_this_level  only  1 feature??

        current_set_of_features[i] = feature_to_add_at_this_level
        print("On level", i, "i added feature ", feature_to_add_at_this_level, "to current set") #display the highest random number --> best feature to print

    return 


