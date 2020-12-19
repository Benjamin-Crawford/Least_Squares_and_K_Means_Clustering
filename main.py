
from keras.datasets import mnist #using keras just to import mnist dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

#helper function to load mnist dataset and print some info about it
def load_mnist():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    return (train_X, train_y), (test_X, test_y)

#helper function to plot a few images and see their labels
def plot_examples(examples,labels):
    for i in range(len(examples)):
        plt.imshow(examples[i],cmap='gray')
        plt.title('Labled: ' + str(labels[i]))
        plt.show()

#takes in a digit and all labels and changes them to either 1 if they match pos_num
#or -1 if the match neg_num this is to set up for one v one least square classifier
#if neg_num is -1 this is a one v all setup
def set_bool_labels(train, test, pos_num, neg_num = -1):
    remove_train = []
    remove_test = []
    for i in range(len(train)):
        if train[i] == pos_num:
            train[i] = 1
        elif train[i] == neg_num or neg_num == -1:
            train[i] = -1
        else:
            remove_train.append(i)

    for i in range(len(test)):
        if test[i] == pos_num:
            test[i] = 1
        elif test[i] == neg_num or neg_num == -1: 
            test[i] = -1
        else:
            remove_test.append(i)

    return train, test, remove_train, remove_test

#takes in mnist dataset images and flattens them into matricies where each 
#row is an image and each column is a different pixel location
#also it normalizes them so that the values are between 0 and 1
def images_2_mat(train,test):
    train = np.true_divide(train.reshape(60000,28*28),255)
    test = np.true_divide(test.reshape(10000,28*28),255)
    print('X_train: ' + str(train.shape))
    print('X_test:  '  + str(test.shape))
    return train, test

#this function takes in the train and test matrices and deletes rows
#in which there are not more than 600 examples that have that pixel as non zero
#this is to reduce training time because we wont have to find coeffecients for pixels
#that are never non zero 
def remove_bad_pix(train,test):
    bad_rows = []
    j = 0
    for row in train.T:
        if np.count_nonzero(row) <= 600:
            bad_rows.append(j)
        j+=1

    train = np.delete(train.T,bad_rows,0).T
    test = np.delete(test.T,bad_rows,0).T

    print('X_train: ' + str(train.shape))
    print('X_test:  '  + str(test.shape))
    return train, test

#this function uses numpys least square solver to solve the least squares problem
#it outputs a vector of weights for each pixel that will be used for classification
def solve_normal_equation(train_x, train_y):
    x_hat = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
    return x_hat

#this class represents a single binary classifier it can be stacked to form either one v one classifiers
#or one v all classifiers
class BinaryClassifier():
    def __init__(self, train_x, train_y, test_x, test_y, true_num, false_num):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.true_num = true_num
        self.false_num = false_num

        #set all labels to match the binary label scheme
        self.bool_train_y, self.bool_test_y, train_remove, test_remove =  set_bool_labels(self.train_y.astype(int), 
                                                                                        self.test_y.astype(int),
                                                                                        self.true_num,
                                                                                        self.false_num)

        #if this classifier is a one v one remove all examples that don't have those two labels
        if len(train_remove) > 0:        
            self.train_x = np.delete(self.train_x,train_remove,0)
            self.train_y = np.delete(self.train_y,train_remove,0)
            self.bool_train_y = np.delete(self.bool_train_y,train_remove,0)

        self.x_hat = solve_normal_equation(self.train_x, self.bool_train_y)
    
    #classify one example at index in test
    def classify_single(self, index):
        return np.matmul(self.x_hat, self.test_x[index])

    #classify all examples in test
    def classify_all(self):
        results = []
        for index in range(self.test_x.shape[0]):
            results.append(np.matmul(self.x_hat, self.test_x[index]))
        return results

class OnevAllClassifier():
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.bin_classifiers = []
        
        #creates ten binary classifiers(one for each digit)
        for i in range(10):
            self.bin_classifiers.append(BinaryClassifier(train_x,train_y,test_x,test_y,i,-1))

    #classifies one example by taking the max of all of the outputs from the binary classifiers
    def classify(self,index):
        results = []
        for classifier in self.bin_classifiers:
            results.append(classifier.classify_single(index))

        return results.index(max(results))

    def classify_all(self):
        results = []
        for index in range(test_x.shape[0]):
            val = self.classify(index)
            results.append(val)

        return results

    #calculates confusion matrix
    def get_confusion_matrix(self):
        errors = 0
        confusion_filepath = "C:\\Users\\ben99\\Desktop\\Homework\\ECE_174\\MiniProject1\\OnevAllClassifierConfusionMatrix.csv"
        self.pred_results = self.classify_all()
        self.true_results = self.test_y
        self.confusion = np.zeros((10,10))

        for i in range(len(self.pred_results)):
            pred_val = int(self.pred_results[i])
            true_val = int(self.true_results[i])
            if not (pred_val == true_val):
                errors = errors + 1
            self.confusion[pred_val][true_val] = self.confusion[pred_val][true_val] + 1

        self.error_rate = (errors/self.test_x.shape[0]) * 100
        print("One V All - Errors: {} | Error Rate: {}%".format(errors,self.error_rate))
        np.savetxt(confusion_filepath,self.confusion,fmt='%d',delimiter=',')

class OnevOneClassifier():
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.bin_classifiers = []

        #stacks binary classfiers with all possible pairings of digits
        for i in range(10):
            for j in range(i+1,10):
                self.bin_classifiers.append(BinaryClassifier(train_x,train_y,test_x,test_y,i,j))

    #classifies one example by running the example through all binary classifiers and checking which
    #digit gets the most votes where a vote is a classfier saying yes it is that number
    def classify(self,index):
        results = np.zeros(10)
        for classifier in self.bin_classifiers:
            this_pos_num = classifier.true_num
            this_neg_num = classifier.false_num
            this_prediction = classifier.classify_single(index)
            if(this_prediction > 0):
                results[this_pos_num] = results[this_pos_num] + 1
            else:
                results[this_neg_num] = results[this_neg_num] + 1

        return list(results).index(max(results))

    def classify_all(self):
        results = []
        for index in range(test_x.shape[0]):
            val = self.classify(index)
            results.append(val)

        return results

    def get_confusion_matrix(self):
        errors = 0
        confusion_filepath = "C:\\Users\\ben99\\Desktop\\Homework\\ECE_174\\MiniProject1\\OnevOneClassifierConfusionMatrix.csv"
        self.pred_results = self.classify_all()
        self.true_results = self.test_y
        self.confusion = np.zeros((10,10))

        for i in range(len(self.pred_results)):
            pred_val = int(self.pred_results[i])
            true_val = int(self.true_results[i])
            if not (pred_val == true_val):
                errors = errors + 1
            self.confusion[pred_val][true_val] = self.confusion[pred_val][true_val] + 1

        self.error_rate = (errors/self.test_x.shape[0]) * 100
        print("One V One - Errors: {} | Error Rate: {}%".format(errors,self.error_rate))
        np.savetxt(confusion_filepath,self.confusion,fmt='%d',delimiter=',')

#this class represents a member of a cluster
class Member():
    def __init__(self,vector,cluster,distance):
        self.vector = vector
        self.cluster = cluster
        self.distance = distance

#this class represents the entire cluster
class Cluster():
    def __init__(self,rep, cluster_num):
        self.cluster_num = cluster_num
        self.rep = rep
        self.cluster_members = []

    #calculates the new representative vector by averaging all vectors in the cluster
    def calc_new_rep(self):
        if len(self.cluster_members) > 0:
            new_rep = np.zeros_like(self.cluster_members[0].vector)
            for member in self.cluster_members:
                new_rep = new_rep + member.vector
            new_rep = new_rep / len(self.cluster_members)
            self.rep = new_rep

    #calculates the sum of the squared distances of all the members of the cluster
    def calc_cluster_distance(self):
        tot_distance = 0
        for member in self.cluster_members:
            member.distance = (np.linalg.norm(self.rep - member.vector) ** 2)
            tot_distance = tot_distance + member.distance
        return tot_distance

    #plots the ten closest members to the cluster and saves the plot at the given filepath
    def plot_closest_members(self,filepath):
        dist_list = []
        for i in range(len(self.cluster_members)):
            dist_list.append(self.cluster_members[i].distance)
        closest_members = np.array(dist_list).argsort()[:10]

        fig = plt.figure(figsize=(15,15))
        ax = []
        columns = 2
        rows = 5
        i = 0
        for i in range(len(closest_members)):
            this_member = self.cluster_members[closest_members[i]]
            cluster_rep = this_member.vector.reshape(28,28,1) * 255
            ax.append(fig.add_subplot(rows, columns, i+1) )
            ax[-1].set_title("Cluster Member " + str(closest_members[i]))
            plt.imshow(cluster_rep, cmap='gray' )
            i = i + 1
        fig.savefig(filepath)

#this class represents the entire K Means Algorithm
class KMeansAlgorithm():
    def __init__(self,x_data,K,min_delta):
        self.x_data = x_data
        self.K = K
        self.min_delta = min_delta
        self.clusters = []
        self.J_clusts = [10**30] #set initial jClust to be very large number
    
    #assigns the initial cluster representatives by randomly selecting them 
    def initial_assignment(self):
        rand_vecs = np.random.randint(0,60000, size=self.K)
        for i in range(len(rand_vecs)):
            self.clusters.append(Cluster(self.x_data[rand_vecs[i]],i))
    
    #sorts the data into the clusters by finding the representative that it is closest to
    def sort_members(self):
        for cluster in self.clusters:
            cluster.cluster_members = [] #clear the members before reassigning them

        for member in self.x_data:
            min_distance = 10**30 #initial assignment is just a very large number
            min_cluster = 0

            for i in range(len(self.clusters)):
                this_distance = (np.linalg.norm(self.clusters[i].rep - member) ** 2)
                if this_distance < min_distance:
                    min_distance = this_distance
                    min_cluster = i

            self.clusters[min_cluster].cluster_members.append(Member(member,min_cluster,min_distance))

    #calculates Jclust by having each cluster calculate its total distance and then normalizing by the
    #number of vectors in the dataset. This method then checks if Jclust has gone down by at least min_delta
    #if not then it will return done which will mean that Jclust has converged.
    def calc_J_clust(self):
        this_j_clust = 0
        done = False
        for cluster in self.clusters:
            this_j_clust = (this_j_clust + (cluster.calc_cluster_distance()/self.x_data.shape[0])) 

        if (self.J_clusts[-1] - this_j_clust) < self.min_delta:
            done = True

        self.J_clusts.append(this_j_clust)

        return done

    #has each cluster recalc its rep
    def reassign_all_reps(self):
        for cluster in self.clusters:
            cluster.calc_new_rep()

    #after the algorithm has been executed this model will plot the trajectory of Jclust
    def plot_all_J_clust(self,filepath):    
        fig, ax = plt.subplots()
        ax.plot(list(range(1,len(self.J_clusts))),self.J_clusts[1:])
        
        ax.set(xlabel='Iterations', ylabel='Jclust',
            title='Value of Jclust at each Iteration')
        ax.grid()

        fig.savefig(filepath)
        plt.show()

    #this function plots all the representatives and then saves it to filepath
    def plot_all_reps(self,filepath):
        fig = plt.figure(figsize=(15, 15))
        ax = []
        columns = 4
        rows = 5
        i = 0
        for cluster in self.clusters:
            cluster_rep = cluster.rep.reshape(28,28,1) * 255
            ax.append(fig.add_subplot(rows, columns, i+1) )
            ax[-1].set_title("Cluster Rep "+str(i))
            plt.imshow(cluster_rep, cmap='gray' )
            i = i + 1
        fig.savefig(filepath)
        plt.show()  # finally, render the plot
    
    #this method plots the 10 closest members to each cluster
    def plot_all_clusters_closest_members(self,filepath):
        i = 0
        for cluster in self.clusters:
            cluster.plot_closest_members(filepath + "_{}.png".format(i))
            i = i + 1

    #this method executes the algorithm by first randomly assigning the cluster representatives
    #then looping between sorting the members and recalculating the reprentatives until Jclust converges
    def execute(self):
        done = False
        self.initial_assignment()
        while(not done):
            self.sort_members()
            done = self.calc_J_clust()
            self.reassign_all_reps()
            print("Iteration {}\nJclust = {}".format(len(self.J_clusts)-1,self.J_clusts[-1]))

#Utility method for data collection about K means algorithm it does multiple runs and collects data on the best and worst one of the set        
def run_several_k_means(x_data,K,P):
    min_delta = 0.5
    min_jclust = 10 ** 30
    max_jclust = 0.1
    k_means_objects = []

 
    for i in range(P):
        print("--------------Run Number {}-----------------".format(i))
        k_means_objects.append(KMeansAlgorithm(x_data,K,min_delta))
        k_means_objects[i].execute()

        
    for run in k_means_objects:
        if run.J_clusts[-1] < min_jclust:
            min_jclust = run.J_clusts[-1]
            min_run = run
        if run.J_clusts[-1] > max_jclust:
            max_jclust = run.J_clusts[-1]
            max_run = run

    min_run.plot_all_J_clust("C:\\Users\\ben99\\Desktop\\Homework\\ECE_174\\MiniProject1\\k_means_plots\\Min_J_Clusts_Plot_K_{}_P_{}.png".format(K,P))
    max_run.plot_all_J_clust("C:\\Users\\ben99\\Desktop\\Homework\\ECE_174\\MiniProject1\\k_means_plots\\Max_J_Clusts_Plot_K_{}_P_{}.png".format(K,P))

    min_run.plot_all_reps("C:\\Users\\ben99\\Desktop\\Homework\\ECE_174\\MiniProject1\\k_means_plots\\Min_Representatives_Plot_K_{}_P_{}.png".format(K,P))
    max_run.plot_all_reps("C:\\Users\\ben99\\Desktop\\Homework\\ECE_174\\MiniProject1\\k_means_plots\\Max_Representatives_Plot_K_{}_P_{}.png".format(K,P))

    min_run.plot_all_clusters_closest_members("C:\\Users\\ben99\\Desktop\\Homework\\ECE_174\\MiniProject1\\k_means_plots\\Min_Closest_Members_Plot_K_{}_P_{}_Cluster".format(K,P))
    max_run.plot_all_clusters_closest_members("C:\\Users\\ben99\\Desktop\\Homework\\ECE_174\\MiniProject1\\k_means_plots\\Max_Closest_Members_Plot_K_{}_P_{}_Cluster".format(K,P))

    return min_run, max_run
            




(train_x, train_y), (test_x, test_y) = load_mnist()
plot_examples(train_x[0:5], train_y[0:5])
train_x, test_x = images_2_mat(train_x,test_x)

# min_run, max_run = run_several_k_means(train_x,10,2)

train_x, test_x = remove_bad_pix(train_x,test_x)
onevall = OnevAllClassifier(train_x,train_y,train_x,train_y)
onevall.get_confusion_matrix()
onevone = OnevOneClassifier(train_x,train_y,train_x,train_y)
onevone.get_confusion_matrix()






