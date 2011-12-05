#!/usr/bin/python

import os
import sys
import math
import time
import random

from optparse import OptionParser

import make_svm_data as make_d

import numpy as np
import milk
import milk.supervised.randomforest as mil_rf
import milk.supervised.multi as mil_multi

from mlrunner import MLRunner

DECODE_DIC = {(0,): 'A', (1,): 'C', (2,): 'G', (3,): 'U'}

###
def leave_one_out(y, x, param=None, n=None):
    results = []
    for i, test in enumerate(zip(y, x)):
        training_y = y[:i] + y[i+1:]
        training_x = x[:i] + x[i+1:]

        training_y = np.array(training_y)
        training_x = np.array([np.array(tx) for tx in training_x])

        learner = mil_rf.rf_learner()
        learner = mil_multi.one_against_one(learner)
        model = learner.train(training_x, training_y)
        result = model.apply(np.array(x[i:i+1][0]))

        results.append((result,) + (test[0], 
                                    make_d.decode(x[i], DECODE_DIC)))
    return results


###
class MilkRunner(MLRunner):

    #
    def __init__(self):
        super(MilkRunner, self).__init__()

        self.cvfunc = leave_one_out
        self.n_cv = None

        pass

    #
    def set_parameters(self, argv):
        options, args = super(MilkRunner, self).set_parameters(argv)
        return options, args

  
        
    #
    def run(self, argv):       
        assert(self.data is not None)

        encode_dic = {'A': [0], 'C': [1], 'G': [2], 'U': [3]}
        recog_rates = []

        
        self.start_logging(['RFOREST',
                            str(int(self.randomize_data)), argv[0]])        
        
        i = 0
        while i < self.n_runs:
            sys.stdout.write('%i ' % i)
            sys.stdout.flush()

            if self.limit_sets:
                new_sets = make_d.merge_multiclasses(self.data, 
                                                     self.set1, self.set2)
                sets = make_d.make_set(new_sets, training_fraction=1.0)
            else:
                sets = make_d.make_set(self.data, training_fraction=1.0)
            train_y, train_x, test_y, test_x = sets

            if self.randomize_data:
                random.shuffle(train_y)
                random.shuffle(test_y)
                pass
        
            print [len(x) for x in sets]

            train_x = [make_d.encode(x, encode_dic) for x in train_x]
            test_x  = [make_d.encode(x, encode_dic) for x in test_x]
       
            t0 = time.clock()
            results = self.cvfunc(train_y, train_x)
            t1 = time.clock()
            print 'Time:', t1 - t0, 
            print 'Remaining:', (self.n_runs-(i+1)) * (t1 - t0)

            cur_acc = super(MilkRunner, self).compute_accuracy(results)
            recog_rates.extend(results)
            total_acc = super(MilkRunner, self).compute_accuracy(recog_rates)

            sys.stdout.write('%.5f,N/A,%.5f\n' % (cur_acc, total_acc))
            sys.stdout.flush()
            self.write_log('%f,%f\n' % (cur_acc, total_acc))
            i += 1
            pass
        
        print 'ACC', super(LIBSVMRunner, self).compute_accuracy(recog_rates)
        self.stop_logging()
        return None    


def main(argv):

    runner = MilkRunner()
    runner.setup(argv)
    # return None
    runner.run(argv)
    
    return None


if __name__ == '__main__': main(sys.argv[1:])

