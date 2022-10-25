#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:32:11 2022

@author: amin2
"""

import torch as pt
import numpy as np
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import random
import argparse
from argparse import Namespace
import os

def log_args(log_dir, log_name, args):
    """ print and save all args """
    if not os.path.exists(log_dir+'/'+log_name):
        os.makedirs(log_dir+'/'+log_name)
    with open(os.path.join(log_dir, 'args_log'), 'w') as f:
        lines = [' {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
        f.writelines(lines)
    for line in lines:
        print(line.rstrip())
        print('-------------------------------------------')

def get_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--log-dir', type=str, default='', help='Directory in which the results are saved.')
    parser.add_argument('--log-name', type=str, default='data', help='Directory in which the results are saved.')
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations of train/test procedure.')
    parser.add_argument('--range-labels', type=int, default=20, help='Maximum number of samples that human could label.')
    ar = parser.parse_args()
    return ar


def preprocess_args(ar):
    if ar.log_dir is None:
        assert ar.log_name is not None

def run():
    configs     =   get_args()    
    main(**vars(configs))


    
    
def main(**kwargs):
    preprocess_args(Namespace(**kwargs))
    log_args(Namespace(**kwargs).log_dir, Namespace(**kwargs).log_name, Namespace(**kwargs))    


    pt.autograd.set_detect_anomaly(True)
    device = pt.device("cpu")
    Size = 1000
    Sizehalf= int(np.floor(Size/2))
    x = pt.rand([Size, 1])
    q = 0;
    qp =0;
    x = pt.sort(x, dim=0).values
    sizexhalf = int(pt.where(x>0.5)[0][0].item())
    y_half = pt.floor(pt.rand([sizexhalf,1] )*2)
    y_other_half = pt.ones([Size-sizexhalf,1] )
    y = pt.vstack((y_half, y_other_half))
    m_half = y_half;
    m_other_half = pt.zeros([Size-sizexhalf,1] )
    m = pt.vstack((m_half, m_other_half))
    # print(y.shape)
    dataloader = pt.hstack((x, y, m))
    # print(dataloader)
    
        
    trainloader=pt.utils.data.DataLoader(dataloader, batch_size=100, shuffle=True, num_workers=8)
    
    class lin_class:
        def __init__(self, q):
            self.q = q;
        def __iter__(self):
            class lin_c:
                def __iter__(self):
                    self.b = 0;
                    return self
                def __next__(self):
    #                 print("Next")
                    # print(self.b)
                    if self.b<1:
                        
                        def lin(b, x):
                            ret = []
                            if  isinstance(x,(list)):
                                ret = np.zeros([len(x), ])
                                # print("It's list with length ", len(x))
                                x = np.array(x)
                                for j in range(len(x)):
                                    
                                    if (x[j]<b):
                                        # print("x =",x, " b=", self.b)
                                        ret[j]=0;
                                    else:
                                        # print("x =",x, " b=", self.b)
                                        ret[j]=1;
                                    # print("Going over x!", j)
                                return ret
                                    # return ret
                            # for j in range(len(x)):
                            if (x<b):
                                # print("x =",x, " b=", self.b)
                                return 0;
                            else:
                                # print("x =",x, " b=", self.b)
                                return 1;
                            # return ret
                        # lin = lambda x: linear(self.b, x)
                        self.b += 1/self.q
                    else:
                        raise StopIteration
                    return self.b, lin;
                def __init__(self, q):
                    self.q = q;
                def __len__(self):
                    return self.q
            return lin_c(self.q)
    
    class vclass:
        def __init__(self):
            self.b_tot = [];
            # print(self.func_dict)
        def append(self,b, func):
            
            self.func=func;
            self.b_tot.append(b)
            # print(self.func_dict)
        def __iter__(self):
            class vc:
                def __init__(self, b_tot, func):
                    self.func=func;
                    self.b_tot= b_tot;
                    self.count =0;
                def __iter__(self):
                    return self;
                def __next__(self):
                    if self.count<len(self.b_tot):
                        self.count +=1;
                    else:
                        # print("Stop Iteration!")
                        raise StopIteration
                    # print(len(self.b_tot), self.count)
                    return self.b_tot[self.count-1], self.func
                def __len__(self):
                    return len(self.b_tot)
            return vc( self.b_tot, self.func);
        
    
    def update_V(version_space, x, y):
        v = iter(version_space)
        fs = vclass();
        # print("Update V")
        for b, f in v:
            if (f(b, x)==y):
                fs.append( b, f);
        return fs;
                        
        
    def is_in_DIS(version_space, x, num_class):
        count = np.zeros([num_class, ])
        # for i in range(num_class):
        v = iter(version_space)
        for b, f in v:
            # if (f(x)==i):
            # print(b, f)
            count[f(b, x)] +=1
                    
                    
        if (sum(count>0)!=1):
            return 1
        # print(count)
        return 0;
    
    
    class generator_uni:
        def draw(self):
            t = np.random.uniform()
            return t
    
    unif = generator_uni()
    
    
    rand_dict = {}
    def y_query(x):
        if (x>0.3):
            return 1;
        elif (x<=0.3):
            if x not in rand_dict.keys():
                rand_dict[x]= 1*(np.random.uniform()>0.5)
            return rand_dict[x];
    def exp_query(x):
        if (x>0.3):
            return 0;
        elif (x<=0.3):
            if x not in rand_dict.keys():
                rand_dict[x]= 1*(np.random.uniform()>0.5)
            return rand_dict[x];
        
    def DoD(version, x_dist, y_query, exp_query, num_class, n_l, n_u, hyp_h, hyp_r):
        TOL = 1e-6
        myclass = version
        i =0
        while (i<n_l):
            
            x = x_dist.draw()
            # print(x)
            if(is_in_DIS(myclass, x, num_class)==1):
                i +=1
                d = int(y_query(x)!=exp_query(x))
                # print(x, y_query(x), exp_query(x), y)
                myclass = update_V(myclass, x, d)
                v = iter( myclass)
                # print(len(v))
            if (len(iter( myclass))==1):
                break
        
        x_data = []
        y_data = []
        for i in range(n_u):
            x = x_dist.draw()
            x_data.append(x)
            y_data.append(y_query(x))
        # print( y_data)
        errmin = 1;
        # plt.scatter(x_data, h(theta_h, x_data).T)
        list_ret = []
        # x_data = np.array(x_data)
        # y_data = np.array(y_data)
        for theta_d, d in myclass:
            # print(d)
            # plt.figure()
            # plt.scatter(x_data, d(theta_d, x_data).T)
            # print("Searching!")
            # print(theta_d)
            for theta_h, h in iter(hyp_h):
                for theta_r, r in iter(hyp_r):
                    errh = 1*(h(theta_h, x_data)!=np.array(y_data))
                    rej = 1-r(theta_r, x_data)
                    nonrej = r(theta_r, x_data)
                    ds = d(theta_d, x_data)
                    err = np.sum(errh*nonrej+rej*ds)/len(x_data)
                    # print(h(theta_h, x_data))
                    # print(err)
                    # print(y_data)
                    # plt.figure()
                    # plt.scatter(x_data, h(theta_h, x_data).T)
                    # plt.scatter(x_data, y_data)
                    if (err<errmin):
                        errmin=err
                        thetah = theta_h
                        thetar = theta_r
                    # break
                    if err<TOL:
                        # print(theta_d)
                        list_ret.append( (theta_h, h, theta_r, r, theta_d, d))
        if len(list_ret)==0:
            print("No perfect answer!")
            for theta_d, d in myclass:
                for theta_h, h in iter(hyp_h):
                    for theta_r, r in iter(hyp_r):
                        errh = 1*(h(theta_h, x_data)!=np.array(y_data))
                        rej = 1-r(theta_r, x_data)
                        nonrej = r(theta_r, x_data)
                        ds = d(theta_d, x_data)
                        err = np.sum(errh*nonrej+rej*ds)/len(x_data)
                        if err-errmin<TOL:
                            # print(theta_d)
                            list_ret.append( (theta_h, h, theta_r, r, theta_d, d))  
        random.shuffle(list_ret)
        return list_ret[0];
    
    def train_ERM(x_dist, y_query, exp_query, num_class, hyp_h, hyp_r, num):
        TOL = 1e-6
        x_data = []
        y_data = []
        exp_data = []
        list_ret = []
        for i in range(num):
            x = x_dist.draw()
            x_data.append(x)
            y_data.append(y_query(x))
            exp_data.append(exp_query(x))
        errt = 10000
        for theta_h, h in iter(hyp_h):
                for theta_r, r in iter(hyp_r):
                    errh = 1*(h(theta_h, x_data)!=np.array(y_data))
                    rej = 1-r(theta_r, x_data)
                    nonrej = r(theta_r, x_data)
                    ds = 1*(np.array(exp_data)!=np.array(y_data))
                    err = np.sum(errh*nonrej+rej*ds)
                    # print(h(theta_h, x_data))
                    # print(err)
                    # print(y_data)
                    # plt.figure()
                    # plt.scatter(x_data, h(theta_h, x_data).T)
                    # plt.scatter(x_data, y_data)
                    
                    if (err<errt):
                        errt=err
                        thetah = theta_h
                        thetar = theta_r
                    # break
                    if err<TOL:
                        # print(theta_d)
                        list_ret.append((theta_h, h, theta_r, r))
        if len(list_ret)!=0:
            random.shuffle(list_ret)
            return list_ret[0];         
        return (thetah, h, thetar, r)
    
    def train_CAL(x_dist, y_query, exp_query, num_class, hyp_h, hyp_r, n_l, n_u):
        TOL = 1e-6
        x_data_u = []
        y_data_u = []
        x_data = []
        y_data = []
        exp_data = []
        list_hr = []
        list_ret = []
        for i in range(n_u):
            x = x_dist.draw()
            
            # if (i<=n_u):
            x_data_u.append(x)
            yq = y_query(x)
            y_data_u.append(yq)
    #         else:
    #             x_data.append(x)
    #             y_data.append(y_query(x))
    #             exp_data.append(exp_query(x))
                
        for theta_h, h in iter(hyp_h):
            for theta_r, r in iter(hyp_r):
                # errh_nl = 1*(h(theta_h, x_data)!=np.array(y_data))
                # nonrej_nl = r(theta_r, x_data)
                errh_nu = 1*(h(theta_h, x_data_u)!=np.array(y_data_u))
                nonrej_nu = r(theta_r, x_data_u)
                err = np.sum(errh_nu*nonrej_nu) 
                # + np.sum(errh_nu*nonrej_nu)
                if err<TOL:
                    list_hr.append((theta_h, h, theta_r, r))
        t = 0            
        # print(len(list_hr))
        while (t<n_l):
            
            # print(t)
            x = x_dist.draw()
            hr = iter(list_hr)
            # print(len(list_hr))
            count = np.zeros([num_class, ])
            
            r_params = []
            for theta_h, h, theta_r, r in hr:
                r_params.append(theta_r)
            if len(set(r_params))==1:
                print("Only 1 rejector is remained", set(r_params))
                break;
            # print(min(r_params))
            # print("Passed finding rejectors. Number of rejectors are "+str(len(set(r_params))))
            hr = iter(list_hr)
            for theta_h, h, theta_r, r in hr:
                # print("HR
                count[r(theta_r, x)] +=1
                if (sum(count>0)!=1):
                    yq = y_query(x)
                    expq = exp_query(x)
                    # print(len(list_hr))
                    iter_hr = iter(list_hr)
                    list_hr = [(theta_hh, hh, theta_rr, rr)for theta_hh, hh, theta_rr, rr in iter_hr  if ((r(theta_rr, x)==0 and yq == expq) or (r(theta_rr, x)==1))]
                    t +=1
                    # print(t, len(list_hr))
                    break;
                    
            # print(count)
        # print("t=", t)
        # print(len(list_hr))
        if len(list_hr)!=0:
            random.shuffle(list_hr)
            # print("Return")
            # print(len(list_hr))
            return list_hr[0];   
        print("No list exists")
        return 0
            
    
    def train_staged_ERM(x_dist, y_query, exp_query, num_class, hyp_h, hyp_r, n_u, n_l):
        myclass = hyp_h
        TOL = 1e-6
        x_data_u = []
        y_data_u = []
        x_data = []
        y_data = []
        exp_data = []
        list_ret = []
        hlist = []
        for i in range(n_u):
            x = x_dist.draw()
            x_data_u.append(x)
            yq = y_query(x)
            y_data_u.append(yq)
            
        # Find the classifiers with minimum error
        errmin = 1;
        for theta_h, h in iter(hyp_h):
            errh = 1*(h(theta_h, x_data_u)!=np.array(y_data_u))
            err = np.sum(errh)/len(x_data_u)
            if (err<errmin):
                errmin = err
                # break
                
        for theta_h, h in iter(hyp_h):
            errh = 1*(h(theta_h, x_data_u)!=np.array(y_data_u))
            err = np.sum(errh)/len(x_data_u)
            if err-errmin<TOL:
                # print(theta_d)
                hlist.append((theta_h, h))
        # print("Length of classifier hypothesis class is ", len(hlist))
        for i in range(n_l):
            x = x_dist.draw()
            x_data.append(x)
            y_data.append(y_query(x))
            exp_data.append(exp_query(x))
                
        errmin = 1
        for theta_h, h in iter(hlist):
                for theta_r, r in iter(hyp_r):
                    errh = 1*(h(theta_h, x_data)!=np.array(y_data))
                    rej = 1-r(theta_r, x_data)
                    nonrej = r(theta_r, x_data)
                    ds = 1*(np.array(exp_data)!=np.array(y_data))
                    err = np.sum(errh*nonrej+rej*ds)/len(x_data)
                    # print(h(theta_h, x_data))
                    # print(err)
                    # print(y_data)
                    # plt.figure()
                    # plt.scatter(x_data, h(theta_h, x_data).T)
                    # plt.scatter(x_data, y_data)
                    
                    if (err<errmin):
                        errmin=err
                    # break
                    if err<TOL:
                        # print(theta_d)
                        list_ret.append((theta_h, h, theta_r, r))
        if len(list_ret)==0:
            for theta_h, h in iter(hlist):
                for theta_r, r in iter(hyp_r):
                    errh = 1*(h(theta_h, x_data)!=np.array(y_data))
                    rej = 1-r(theta_r, x_data)
                    nonrej = r(theta_r, x_data)
                    ds = 1*(np.array(exp_data)!=np.array(y_data))
                    err = np.sum(errh*nonrej+rej*ds)/len(x_data)
                    if err-errmin<=TOL:
                        # print(theta_d)
                        list_ret.append((theta_h, h, theta_r, r))
        random.shuffle(list_ret)
        return list_ret[0];  
        # return (thetah, h, thetar, r)
                
        
        
            
    def test_def(x_dist, y_query, exp_query, theta_h, h, theta_r, r, num):
        x_data = []
        y_data = []
        m_data = []
        for i in range(num):
            x = x_dist.draw()
            x_data.append(x)
            y = y_query(x)
            m = exp_query(x)
            # if (x<0.3):
            #     print(m, y)
            y_data.append(y)
            m_data.append(m)
        errh = 1*(h(theta_h, x_data)!=np.array(y_data))
        # print(np.sum(errh)/num)
        rej = 1-r(theta_r, x_data)
        nonrej = r(theta_r, x_data)
        ds = 1*(np.array(m_data)!=np.array(y_data))
        # print(m_data, y_data, ds)
        err = np.sum(errh*nonrej+rej*ds)
        return err/num
    
    
    num_class = 2
    range_labels = Namespace(**kwargs).range_labels
    ITERS = Namespace(**kwargs).iters
    test_err =np.zeros([range_labels,])
    test_err_ERM = np.zeros([range_labels,])
    test_err_CAL =np.zeros([range_labels,])
    test_err_staged_ERM = np.zeros([range_labels,])
    n_test = 1000
    n_u = 100
    quant_models = 100
    for t in range(1, range_labels+1):
        print("T= ", t)
        for p in range(ITERS):
            hclass = lin_class(quant_models)
            rclass = lin_class(quant_models)
            theta_h, h, theta_r, r = train_ERM(unif, y_query, exp_query, num_class, hclass, rclass, t)
            # print(theta_h, theta_r, theta_d)
            test_err_ERM[t-1] += test_def(unif, y_query, exp_query, theta_h, h, theta_r, r,  n_test)
        for p in range(ITERS):
            dclass = lin_class(quant_models)
            hclass = lin_class(quant_models)
            rclass = lin_class(quant_models)
            theta_h, h, theta_r, r, theta_d, d = DoD(dclass, unif, y_query, exp_query, num_class, t, n_u, hclass, rclass)
            # print(theta_h, theta_r, theta_d)
            test_err[t-1] += test_def(unif, y_query, exp_query, theta_h, h, theta_r, r,  n_test)
        
        for p in range(ITERS):
            
            hclass = lin_class(quant_models)
            rclass = lin_class(quant_models)
            theta_h, h, theta_r, r = train_staged_ERM(unif, y_query, exp_query, num_class, hclass, rclass, n_u, t)
            
            # print(theta_h, theta_r, theta_d)
            test_err_staged_ERM[t-1] += test_def(unif, y_query, exp_query, theta_h, h, theta_r, r,  n_test)
        for p in range(ITERS):
            hclass = lin_class(quant_models)
            rclass = lin_class(quant_models)
            theta_h, h, theta_r, r = train_CAL( unif, y_query, exp_query, num_class, hclass, rclass, t, n_u)
            # print(theta_h, h, theta_r, r)
            # print(theta_h, theta_r, theta_d)
            test_err_CAL[t-1] += test_def(unif, y_query, exp_query, theta_h, h, theta_r, r,  n_test)    
        
    
    np.savez(Namespace(**kwargs).log_dir+'/'+Namespace(**kwargs).log_name+'/'+"DataCAL.npy", range_labels=range_labels, test_err_CAL=test_err_CAL, test_err_staged_ERM=test_err_staged_ERM)
    np.savez(Namespace(**kwargs).log_dir+'/'+Namespace(**kwargs).log_name+'/'+"Data.npy", range_labels=range_labels, test_err=test_err, test_err_ERM=test_err_ERM)

if __name__ == '__main__':
    run()
