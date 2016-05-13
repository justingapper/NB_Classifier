# file: spam.py
"""
@author: jgapper
"""

# import packages
import numpy as np
import pandas as pd

# import data
data = pd.read_csv('C:\Users\jgapper\Desktop\CSDS\CS530\Assignment1\spamdata_binary.csv', header=None)
labels = pd.read_csv('C:\Users\jgapper\Desktop\CSDS\CS530\Assignment1\spamlabels.csv', header = None)
groups = pd.read_csv('C:\Users\jgapper\Desktop\CSDS\CS530\Assignment1\group.csv', index_col = None, header = None)


class spam(object):
    def __init__(self):
        # initiate variables
        self.data = data # raw spam data
        self.labels = labels # raw labels
        self.labels.columns = ['labels']
        self.data['labels'] = self.labels # add labels to spam data
        self.groups = groups # groups for cross validation
        self.sp_count = float # spam count
        self.hm_count = float # ham count
        self.spam_data = pd.DataFrame # spam df
        self.ham_data = pd.DataFrame # ham df
        self.spam_subset = pd.DataFrame # spam subset
        self.ham_subset = pd.DataFrame # ham subset
        self.likelihood_sp = pd.DataFrame # likelihood
        self.bayes_score = pd.DataFrame # bayes measure
        self.acc_scores = []

    def rand(self, data):
        # randomize data
        self.data['rand'] = np.random.choice(range(0,10000), data.shape[0])
        self.data = self.data.sort_index(by='rand', ascending=0)
        self.data = self.data.reset_index()

    def fold(self, data):
        # Add groups for cross validation
        self.groups.columns= ['groups']
        self.data['groups']= self.groups

    def train(self, dat):
        # train method, calc bayes spam propensity
        self.spam_data = self.dat[self.dat['labels']==1] # get spam data
        self.ham_data = self.dat[self.dat['labels']==0] # get ham data
        self.sp_count = (len(self.spam_data)) # number of spam
        self.hm_count = (len(self.ham_data)) # number of ham
        self.tot = float(len(self.dat)) # total number of observations
        self.prior_sp = (self.sp_count / self.tot) # calc prior
        self.prior_hm = (1- self.prior_sp) # calc ham percent
        self.spam_subset = self.spam_data.iloc[:,0:58].apply(pd.value_counts) # spam subset
        self.ham_subset = self.ham_data.iloc[:,0:58].apply(pd.value_counts) # ham subset

        # spam likelihood
        self.likelihood_sp = (self.spam_subset.iloc[1,:]/self.sp_count) # calc likelihood spam
        self.evidence = (self.spam_subset.iloc[1,:].add(self.ham_subset.iloc[1,:])/self.tot) # calc evidence spam
        self.num_sp = self.likelihood_sp * self.prior_sp # calculate numerator spam
        self.bayes_sp = self.num_sp.div(self.evidence) # calculate bayes spam propensity by attribute

        # ham likelihood
        self.likelihood_hm = (self.ham_subset.iloc[1,:]/self.hm_count) # calc likelihood ham
        self.num_hm = self.likelihood_hm * self.prior_hm # calculate numerator ham
        self.bayes_hm = self.num_hm.div(self.evidence) # calculate bayes ham propensity by attribute

    def test(self, dat2):
        # test method
        self.score_sp = self.dat2.iloc[:,0:58].mul(self.bayes_sp) # calc score by multiplying out bayes propensity by attribute, by observation
        self.score_sp['labels']=0
        self.max_score_sp = self.score_sp.max(1) # calc spam max likelihood score
        self.score_hm = self.dat2.iloc[:,0:58].mul(self.bayes_hm)
        self.score_hm['labels']=0
        self.max_score_hm = self.score_hm.max(1) # calc ham max likelihood score
        self.dat2['max_score_sp'] = self.max_score_sp # assign spam max likelihood
        self.dat2['max_score_hm'] = self.max_score_hm # assign ham max likelihood
        self.dat2.loc[self.dat2['max_score_sp'] > self.dat2['max_score_hm'], 'class'] = 1 # apply class based on maximum likelihoods
        self.dat2.loc[self.dat2['max_score_sp'] <= self.dat2['max_score_hm'], 'class'] = 0
        self.dat2.loc[self.dat2['class']==self.dat2['labels'], 'correct_class'] = 1 # calculate correct classifications
        self.dat2.loc[self.dat2['class']!=self.dat2['labels'], 'correct_class'] = 0
        self.corr_count = len(self.dat2[self.dat2['correct_class']==1]) # sum correct classifications
        self.acc = self.corr_count/float(len(self.dat2)) # calc accuracy

    def cross(self, data):
        # cross validation loop
        i=1
        for i in range(1,10):
            self.dat = self.data[self.data['groups']!=i] # train on sets not in group i
            sp.train(self.dat)
            self.dat2 = self.data[self.data['groups']==i] # test on setst equal to group i
            sp.test(self.dat2)
            #print(self.acc)
            self.acc_scores.append(self.acc)
            
    def pr(self, data):
        # display results
        self.acc_ave = sum(self.acc_scores)/float(len(self.acc_scores))
        print "The accuracy scores for each of the folds: \n %s \n" % self.acc_scores
        print "The average 10-fold cross-validation accuracy: \n %s \n" % self.acc_ave
        print "Warnings..."

        
sp = spam()
sp.rand(sp.data)
sp.fold(sp.data)
sp.cross(sp.data)
sp.pr(data)