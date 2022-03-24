
import numpy as np
from sklearn.covariance import log_likelihood

class NaiveBayes(object):
    """ Bernoulli Naive Bayes model
    
    ***DO NOT CHANGE the following attribute names (to maintain autograder compatiblity)***
    
    @attrs:
        n_classes:    the number of classes
        attr_dist:    a 2D (n_classes x n_attributes) NumPy array of the attribute distributions
        label_priors: a 1D NumPy array of the priors distribution
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes model with n_classes. """
        self.n_classes = n_classes
        self.attr_dist = None
        self.label_priors = None

    def train(self, X_train, y_train):
        """ Trains the model, using maximum likelihood estimation.
        @params:
            X_train: a 2D (n_examples x n_attributes) numpy array
            y_train: a 1D (n_examples) numpy array
        @return:
            a tuple consisting of:
                1) a 2D numpy array of the attribute distributions
                2) a 1D numpy array of the priors distribution
        """
        # get attr distribution and the priors
        # number instances over total number instances
        # use Y train for priors, for attr distrib, use X train when y train = 0
    
        # num_ones = np.sum(y_train == 1) #get number of 1s
        # prob = (num_ones / len(y_train)) #prob for label 1

        # label_ones = X_train[y_train == 1] #index X_train -> gives data points with label = 1

        # array of just the priors
        #print(X_train.shape)
        # arr = np.zeros([self.n_classes])
        class_freqs = np.zeros([self.n_classes])
        distribs = np.zeros((self.n_classes, X_train.shape[1]))

        for i in range(self.n_classes):

            y_at_class_i = np.count_nonzero(y_train == i) + 1
            class_freqs[i] = y_at_class_i
            
            # prior_prob_i = y_at_class_i / (len(y_train) + self.n_classes)
            # arr[i] = prior_prob_i

            #calc attr distrib
            index = np.argwhere(y_train == i) #index of class
            attr_at_class_i = (X_train[index]) #checks each row for condition
            distribs[i] = np.sum(attr_at_class_i, axis=0) #distribs means count (freq of attr)



            # attr_distr = (np.sum(attr_at_class_i, axis=0) + 1) / (y_at_class_i + (self.n_classes-1))
            # # attr_sum = np.sum(attr_at_class_i) + 1
            # # attr_distrib = ((attr_at_class_i) / len(X_train[i]) + len (X_train[i][y_train == i]))
            # distribs[i] = attr_distr
        distribs = np.transpose(distribs)
        divisor = np.array([class_freqs,] * X_train.shape[1]) + 2 #elt y times each freq by # attrs

        # attr_distr = (np.sum(attr_at_class_i, axis=0) + 1) / (y_at_class_i + (self.n_classes-1))
        attr_distr = (distribs + 1) / divisor
        # distribs = attr_distr
        prior_distrib = (class_freqs + 1) / (len(y_train) + self.n_classes)
        self.attr_dist = attr_distr #should 2 by num_attr array
        self.label_priors = prior_distrib
        return attr_distr,prior_distrib

    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.
            Remember to convert to log space to avoid overflow/underflow
            errors!

        @params:
            inputs: a 2D NumPy array containing inputs
        @return:
            a 1D numpy array of predictions
        """
        predictions = np.zeros(len(inputs))

        for i in range(len(inputs)):
            example = inputs[i].reshape((-1,1)) #gets example and makes column vector
            zeros_flipped = np.where(example == 0, 1 - self.attr_dist, self.attr_dist) #checks for zeros -> returns array of booleans

            # take into log space
            log_probabilities = np.log(zeros_flipped)
            log_probabilities = np.exp(log_probabilities.sum(axis=0))
            log_likelihood = self.label_priors * log_probabilities #array of size num classes

            #get the max -> max index of log probabilities, which is the class we return as prediction
            maximum_index = np.argmax(log_likelihood)
            predictions[i] = maximum_index
            
        return predictions
        # posterior = joint/joint.sum  -> log likelihood
        

    def accuracy(self, X_test, y_test):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            X_test: a 2D numpy array of examples
            y_test: a 1D numpy array of labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test) #ratio of number of correct matches

    def print_fairness(self, X_test, y_test, x_sens):
        """ 
        ***DO NOT CHANGE what we have implemented here.***
        
        Prints measures of the trained model's fairness on a given dataset (data).

        For all of these measures, x_sens == 1 corresponds to the "privileged"
        class, and x_sens == 0 corresponds to the "disadvantaged" class. Remember that
        y == 1 corresponds to "good" credit. 

        @params:
            X_test: a 2D numpy array of examples
            y_test: a 1D numpy array of labels
            x_sens: a numpy array of sensitive attribute values
        @return:

        """
        predictions = self.predict(X_test)

        # Disparate Impact (80% rule): A measure based on base rates: one of
        # two tests used in legal literature. All unprivileged classes are
        # grouped together as values of 0 and all privileged classes are given
        # the class 1. . Given data set D = (S,X,Y), with protected
        # attribute S (e.g., race, sex, religion, etc.), remaining attributes X,
        # and binary class to be predicted Y (e.g., “will hire”), we will say
        # that D has disparate impact if:
        # P[Y^ = 1 | S != 1] / P[Y^ = 1 | S = 1] <= (t = 0.8). 
        # Note that this 80% rule is based on US legal precedent; mathematically,
        # perfect "equality" would mean

        di = np.mean(predictions[np.where(x_sens==0)])/np.mean(predictions[np.where(x_sens==1)])
        print("Disparate impact: " + str(di))

        # Group-conditioned error rates! False positives/negatives conditioned on group
        
        pred_priv = predictions[np.where(x_sens==1)]
        pred_unpr = predictions[np.where(x_sens==0)]
        y_priv = y_test[np.where(x_sens==1)]
        y_unpr = y_test[np.where(x_sens==0)]

        # s-TPR (true positive rate) = P[Y^=1|Y=1,S=s]
        priv_tpr = np.sum(np.logical_and(pred_priv == 1, y_priv == 1))/np.sum(y_priv)
        unpr_tpr = np.sum(np.logical_and(pred_unpr == 1, y_unpr == 1))/np.sum(y_unpr)

        # s-TNR (true negative rate) = P[Y^=0|Y=0,S=s]
        priv_tnr = np.sum(np.logical_and(pred_priv == 0, y_priv == 0))/(len(y_priv) - np.sum(y_priv))
        unpr_tnr = np.sum(np.logical_and(pred_unpr == 0, y_unpr == 0))/(len(y_unpr) - np.sum(y_unpr))

        # s-FPR (false positive rate) = P[Y^=1|Y=0,S=s]
        priv_fpr = 1 - priv_tnr 
        unpr_fpr = 1 - unpr_tnr 

        # s-FNR (false negative rate) = P[Y^=0|Y=1,S=s]
        priv_fnr = 1 - priv_tpr 
        unpr_fnr = 1 - unpr_tpr

        print("FPR (priv, unpriv): " + str(priv_fpr) + ", " + str(unpr_fpr))
        print("FNR (priv, unpriv): " + str(priv_fnr) + ", " + str(unpr_fnr))
    
    
        # #### ADDITIONAL MEASURES IF YOU'RE CURIOUS #####

        # Calders and Verwer (CV) : Similar comparison as disparate impact, but
        # considers difference instead of ratio. Historically, this measure is
        # used in the UK to evalutate for gender discrimination. Uses a similar
        # binary grouping strategy. Requiring CV = 1 is also called demographic
        # parity.

        cv = 1 - (np.mean(predictions[np.where(x_sens==1)]) - np.mean(predictions[np.where(x_sens==0)]))

        # Group Conditioned Accuracy: s-Accuracy = P[Y^=y|Y=y,S=s]

        priv_accuracy = np.mean(predictions[np.where(x_sens==1)] == y_test[np.where(x_sens==1)])
        unpriv_accuracy = np.mean(predictions[np.where(x_sens==0)] == y_test[np.where(x_sens==0)])

        return predictions
