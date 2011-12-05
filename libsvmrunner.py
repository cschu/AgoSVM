#!/usr/bin/python

import os
import sys
import math
import time
import random

from optparse import OptionParser

import make_svm_data as make_d
import svm_functions as svmfun

LIBSVM_PATH = '/home/schudoma/tools/libsvm-3.1/python'
if not os.path.exists(LIBSVM_PATH):
    sys.stderr.write('Missing LIBSVM_PATH. Aborting.\n')
    sys.exit(1)    
sys.path.append(LIBSVM_PATH)

import svmutil
import svm

from mlrunner import MLRunner

###
class LIBSVMRunner(MLRunner):

    #
    def __init__(self):
        super(LIBSVMRunner, self).__init__()
        self.c_range = -5, 15, 2
        self.gamma_range = 3, -15, -2
        self.kernel_type = 'LINEAR'

        self.svmparam = svm.svm_parameter('-b 1')
        self._optparser.add_option('-k', '--kernel', dest='kernel_type')

        self.cvfunc = svmfun.leave_one_out
        self.n_cv = None

        pass

    #
    def set_parameters(self, argv):
        options, args = super(LIBSVMRunner, self).set_parameters(argv)
        if options.kernel_type is None:
            self.kernel_type = 'RBF'
            self.svmparam.kernel_type = svm.RBF
        elif options.kernel_type == 'LINEAR':
            self.svmparam.kernel_type = svm.LINEAR
            self.gamma_range = 1, 0, -2

        return options, args

   

        
    #
    def run(self, argv):       
        assert(self.data is not None)

        param_grid = {}
        results = []
        sum_acc = 0

        self.start_logging([self.kernel_type, 
                            str(int(self.randomize_data)), argv[0]])        
        
        i = 0
        while i < self.n_runs:
            sys.stdout.write('%i ' % i)
            sys.stdout.flush()

            if self.limit_sets:
                new_sets = make_d.merge_multiclasses(self.data, self.set1, self.set2)
                sets = make_d.make_set(new_sets, training_fraction=0.75)
            else:
                sets = make_d.make_set(self.data, training_fraction=0.75)
            train_y, train_x, test_y, test_x = sets

            if self.randomize_data:
                random.shuffle(train_y)
                random.shuffle(test_y)
                pass
        
            print [len(x) for x in sets]

            train_x = [make_d.encode(x, make_d.encode_dic) for x in train_x]
            test_x  = [make_d.encode(x, make_d.encode_dic) for x in test_x]
       
            t0 = time.clock()
            self.svmparam = svmfun.grid_search(train_y, train_x, self.svmparam,
                                               self.cvfunc, self.n_cv, 
                                               self.c_range, self.gamma_range)
            t1 = time.clock()
            print 'Time:', t1 - t0, 
            print 'Remaining:', (self.n_runs-(i+1)) * (t1 - t0)

            problem = svm.svm_problem(train_y, train_x)
            model = svmutil.svm_train(problem, self.svmparam, '-q')
            result = svmutil.svm_predict(test_y, test_x, model, '-b 1')

            cur_result = zip(result[0], test_y)
            cur_acc = super(LIBSVMRunner, self).compute_accuracy(cur_result)        
            results.extend(cur_result)
            total_acc = super(LIBSVMRunner, self).compute_accuracy(results)

            sum_acc += cur_acc
            mean_acc = sum_acc/(i+1)
            sys.stdout.write('%.5f %.5f %.5f\n' % \
                                 (cur_acc, mean_acc, total_acc))
            sys.stdout.flush()

            self.write_log('%f,%f,%f\n' % (cur_acc, mean_acc, total_acc))
            i += 1
            pass
        
        print 'ACC', super(LIBSVMRunner, self).compute_accuracy(results)
        self.stop_logging()
        return None    


def main(argv):

    runner = LIBSVMRunner()
    runner.setup(argv)
    # return None
    runner.run(argv)
    
    return None


if __name__ == '__main__': main(sys.argv[1:])

