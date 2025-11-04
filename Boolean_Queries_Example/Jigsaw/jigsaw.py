import numpy as np
from tqdm import tqdm
import time
from scipy.linalg import blas
import duckdb
class JigsawAttacker:
    ### implementations of Jigsaw and RSA
    def __init__(self,sim_M,real_M,sim_F=None,real_F=None,\
            no_F=False,baseRec = 50,confRec=25,refinespeed = 15,\
            alpha=0.5,beta=0.4,countermeasure_params={"alg":None},real_doc_num=None,refinespeed_exp=None):
        #print("sim_M",sim_M.shape)

        self.sim_M = sim_M
        self.real_M = real_M
        self.real_V = np.sum(real_M,axis=1)
        self.real_V = self.real_V/np.sum(self.real_V)
        self.sim_V = np.sum(sim_M,axis=1)
        self.sim_V = self.sim_V/np.sum(self.sim_V)


        self.sim_F = sim_F
        self.real_F = real_F
        self.no_F = no_F
        self.BaseRec = baseRec
        self.ConfRec = confRec
        self.refinespeed = refinespeed
        self.refinespeed_exp = refinespeed_exp
        self.alpha = alpha
        self.beta = beta
        self.real_doc_num = real_doc_num
        self.tdid_2_kwsid = {}
        self.tdid_2_kwsid_step1 = {}
        self.tdid_2_kwsid_step2 = {}
        self.tdid_2_kwsid_step3 = {}
        self.unrec_td_set = set([i for i in range(len(self.real_M))])
        self.id_known_kws=None
        self.id_queried_kws=None
    def attack_step_1(self):
        #Jigsaw Step1:Locating and recovering the distinctive queries by Volume and/or Frequency
        D_FV = self.calculate_dVF()
        id_Diff = []
        for i in range(len(D_FV)):
            id_Diff.append([i,D_FV[i]])
        id_Diff.sort(key = lambda x:x[1],reverse=True)
        top = id_Diff[:self.BaseRec]
        top_td_list = [i[0] for i in top]
        
        tdid_2_kwsid = self.recover_by_VF(top_td_list)
        self.tdid_2_kwsid_step1 = tdid_2_kwsid
    def attack_step_2(self):
        #Jigsaw Step2:Verify by co-occurance
        tdid_2_kwsid = self.verify_by_M()
        self.tdid_2_kwsid.update(tdid_2_kwsid)
        self.tdid_2_kwsid_step2 = tdid_2_kwsid
        self.unrec_td_set = self.unrec_td_set - set(tdid_2_kwsid.keys())
    
    def attack_step_3(self):
        #Jigsaw Step3:Using co-occurance to recover remaining queries
        while(len(self.unrec_td_set)>0):
            paired_td = list(self.tdid_2_kwsid.keys())
            paired_kw = [self.tdid_2_kwsid[i] for i in paired_td]
            unpaired_kw = list(set([i for i in range(len(self.sim_M))]) - set(paired_kw))
            un_td_list = list(self.unrec_td_set)
            
            M = self.real_M[un_td_list]
            M_ = self.sim_M[unpaired_kw]
            M = M/M.sum(axis = 1).reshape((len(M),1))
            M_ = M_/M_.sum(axis = 1).reshape((len(M_),1))

            # sim_F = self.sim_F[unpaired_kw]
            # real_F = self.real_F[un_td_list]
            sim_V = self.sim_V[unpaired_kw]
            real_V = self.real_V[un_td_list]

            Certainty = []
            for i in range(len(M)):
                S = self.alpha*np.abs(real_V[i]-sim_V) #+(1-self.alpha)*(np.abs(real_F[i]-sim_F))
                score = -np.log(self.beta*np.linalg.norm(M[i]-M_,axis=1)+(1-self.beta)*(S))
                score = sorted(score,reverse=True)
                certainty = score[0]-score[1]
                Certainty.append([i,certainty])
            Certainty.sort(key = lambda x:x[1],reverse=True)
            if len(Certainty)<self.refinespeed:
                top_td = [Certainty[i][0] for i in range(len(Certainty))]
            else:
                top_td = [Certainty[i][0] for i in range(int(self.refinespeed))]
            tdid_2_kwsid = {}
            for i in range(len(top_td)):
                kw = np.argmax(-np.log(np.linalg.norm(M[top_td[i]]-M_,axis=1)))
                tdid_2_kwsid[un_td_list[top_td[i]]]= unpaired_kw[kw]
            self.tdid_2_kwsid.update(tdid_2_kwsid)
            self.tdid_2_kwsid_step3.update(tdid_2_kwsid)
            self.unrec_td_set = self.unrec_td_set - set(tdid_2_kwsid.keys())
            if self.refinespeed_exp == True:
                self.refinespeed = self.refinespeed * 1.1
        return self.tdid_2_kwsid

    def RSA(self):
        ### implementations of RSA
        while(len(self.unrec_td_set)>0):
            paired_td = list(self.tdid_2_kwsid.keys())
            paired_kw = [self.tdid_2_kwsid[i] for i in paired_td]
            unpaired_kw = list(set([i for i in range(len(self.sim_M))]) - set(paired_kw))
            un_td_list = list(self.unrec_td_set)
            
            M = self.real_M[un_td_list][:,paired_td]
            M_ = self.sim_M[:,paired_kw]
           
            Certainty = []
            for i in range(len(M)):
                score = -np.log(np.linalg.norm(M[i]-M_,axis=1))
                score = sorted(score,reverse=True)
                certainty = score[0]-score[1]
                Certainty.append([i,certainty])
            Certainty.sort(key = lambda x:x[1],reverse=True)
            if len(Certainty)<self.refinespeed:
                top_td = [Certainty[i][0] for i in range(len(Certainty))]
            else:
                top_td = [Certainty[i][0] for i in range(int(self.refinespeed))]
            tdid_2_kwsid = {}
            for i in range(len(top_td)):
                kw = np.argmax(-np.log(np.linalg.norm(M[top_td[i]]-M_,axis=1)))
                tdid_2_kwsid[un_td_list[top_td[i]]]= kw#(paired_kw+unpaired_kw)[kw]
            self.tdid_2_kwsid.update(tdid_2_kwsid)
            self.tdid_2_kwsid_step3.update(tdid_2_kwsid)
            self.unrec_td_set = self.unrec_td_set - set(tdid_2_kwsid.keys())
            if self.refinespeed_exp == True:
                self.refinespeed = self.refinespeed * 1.1
        return self.tdid_2_kwsid

    def calculate_dVF(self):
        td_nb = len(self.real_M)
        D_FV = np.zeros(td_nb)
        for i in range(td_nb):
            # d_F = np.abs(self.real_F-self.real_F[i])
            d_V = np.abs(self.real_V-self.real_V[i])
            if self.no_F != True:
                d_FV = self.alpha*d_V #+(1-self.alpha)*d_F
            else:
                d_FV = d_V
            d_FV[i] = float("inf")
            D_FV[i] = np.min(d_FV)
        return D_FV
    def recover_by_VF(self,td_list):
        # using volume and frequency to recover queries in td_list
        tmp_pair = {}
        for i in range(len(td_list)):
            s1 = np.abs(self.sim_V-self.real_V[td_list[i]])
            if self.no_F != True:
                # s2 = np.abs(self.sim_F-self.real_F[td_list[i]])
                s = self.alpha*s1 #+ (1-self.alpha)*s2
            else:
                s = s1
            tmp_pair[td_list[i]] = np.argmin(s)
        return tmp_pair
    def verify_by_M(self):
        # using co-occurance matrix to verify paired queries
        tmp_pair = self.tdid_2_kwsid_step1.copy()
        nb = len(tmp_pair)
        pair = tmp_pair.copy()
        pair_list = []
        for i in tmp_pair.keys():
            pair_list.append([i,tmp_pair[i]])
        td = [i[0] for i in pair_list]
        kw = [i[1] for i in pair_list]
        td_M = self.real_M[td]
        kw_M = self.sim_M[kw]
        td_M  = td_M/td_M.sum(axis = 1).reshape((nb,1))
        kw_M  = kw_M/kw_M.sum(axis = 1).reshape((nb,1))
        Dis = []
        for i in range(nb):
            Dis.append(np.linalg.norm(td_M[i]-kw_M[i]))

        flag = sorted(Dis,reverse= True)[self.BaseRec-self.ConfRec]
        for i in range(len(Dis)):
            if Dis[i]>flag:
                del(pair[pair_list[i][0]])
        return pair
