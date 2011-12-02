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

C_RANGE = -5, 15, 2
GAMMA_RANGE = 3, -15, -2
N_RUNS = 1000

TIMESTAMP = ''
KERNEL_TYPE = 'LINEAR'
RANDOMIZE_DATA = False
SET1 = None
SET2 = None

def init(argv):
    global TIMESTAMP
    TIMESTAMP = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())            
    parser = OptionParser()
    parser.add_option('-k', '--kernel', dest='kernel_type')
    parser.add_option('-n', type='int', dest='n_runs')
    parser.add_option('--random', dest='randomize_data', action='store_true')
    parser.add_option('--s1', dest='set_1')
    parser.add_option('--s2', dest='set_2')
    options, args = parser.parse_args(argv)

    global KERNEL_TYPE
    if options.kernel_type is not None:
        KERNEL_TYPE = options.kernel_type
    global RANDOMIZE_DATA
    if options.randomize_data is not None:
        RANDOMIZE_DATA = options.randomize_data
    global N_RUNS
    if options.n_runs is not None:
        N_RUNS = options.n_runs
    global SET1
    SET1 = [float(v) for v in options.set_1.split(':')]
    global SET2
    SET2 = [float(v) for v in options.set_2.split(':')]

    return None

###
def compute_accuracy(results):
    return sum(map(float, map(lambda x: x[0]==x[1], results)))/len(results)


###
def main(argv):

    global C_RANGE
    global GAMMA_RANGE
    global SET1
    global SET2

    i = 0
    param_grid = {}
    results = []
    sum_acc = 0

    init(argv[1:])    
    print SET1, SET2

    fn = argv[0]
    dataset = make_d.read_data(open(fn))
    dataset = make_d.assign_classes(dataset)
    data = make_d.prepare_data(dataset)
    print data.keys(), [len(v) for v in data.values()]

    param = svm.svm_parameter('-b 1')
    if KERNEL_TYPE == 'LINEAR':
        param.kernel_type = svm.LINEAR
        GAMMA_RANGE = 1, 0, -2
    else:
        param.kernel_type = svm.RBF

    cvfunc = svmfun.leave_one_out
    n_cv = None

    limit_sets = not SET1 is None and not SET2 is None

    outfile = os.path.basename(fn)
    outfile = outfile.replace('.fasta', '')
    outfile = outfile.replace('.fas', '')
    if limit_sets:
        outfile = ''.join(map(str, map(int, SET1))) + 'vs'
        outfile += ''.join(map(str, map(int, SET2)))

    log_name = '%s-%s-%i-%s.csv' % (TIMESTAMP, 
                                    KERNEL_TYPE,
                                    int(RANDOMIZE_DATA),
                                    outfile)
    logfile = open(log_name, 'w')                                    

    while i < N_RUNS:
        sys.stdout.write('%i ' % i)
        sys.stdout.flush()

        if limit_sets:
            new_sets = make_d.merge_multiclasses(data, SET1, SET2)
            sets = make_d.make_set(new_sets, training_fraction=0.75)
        else:
            sets = make_d.make_set(data, training_fraction=0.75)
        train_y, train_x, test_y, test_x = sets

        if RANDOMIZE_DATA:
            random.shuffle(train_y)
            random.shuffle(test_y)
            pass
        
        print [len(x) for x in sets]

        train_x = [make_d.encode(x, make_d.encode_dic) for x in train_x]
        test_x  = [make_d.encode(x, make_d.encode_dic) for x in test_x]
       
        t0 = time.clock()
        param = svmfun.grid_search(train_y, train_x, param,
                                   cvfunc, n_cv, C_RANGE, GAMMA_RANGE)
        t1 = time.clock()
        print 'Time:', t1 - t0, 'Remaining:', (N_RUNS-(i+1)) * (t1 - t0)

        problem = svm.svm_problem(train_y, train_x)
        model = svmutil.svm_train(problem, param, '-q')
        result = svmutil.svm_predict(test_y, test_x, model, '-b 1')

        cur_result = zip(result[0], test_y)
        cur_acc = compute_accuracy(cur_result)        
        results.extend(cur_result)
        total_acc = compute_accuracy(results)

        sum_acc += cur_acc
        mean_acc = sum_acc/(i+1)
        sys.stdout.write('%.5f %.5f %.5f\n' % (cur_acc, mean_acc, total_acc))
        sys.stdout.flush()

        logfile.write('%f,%f,%f\n' % (cur_acc, mean_acc, total_acc))
        logfile.flush()

        i += 1
        pass

    print 'ACC', compute_accuracy(results)
    logfile.close()

    return None    

if __name__ == '__main__': main(sys.argv[1:])

