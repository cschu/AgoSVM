#!/usr/bin/python

'''
Created on Nov 30, 2011

@author: Chris
'''

import os
import sys
import math
import time
import copy
import random

from optparse import OptionParser

import make_svm_data as make_d

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

###
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
    if not options.set_1 is None:
	SET1 = [float(v) for v in options.set_1.split(':')]
    global SET2
    if not options.set_2 is None:
    	SET2 = [float(v) for v in options.set_2.split(':')]

    return None

###
def grid_search(y, x, param, grid, cv_func, n, c_range, gamma_range):
    cstart, cend, cstep = c_range
    gstart, gend, gstep = gamma_range
    # t0 = time.clock()
    for c in xrange(cstart, cend + 1, cstep):
        param.C = 2.0 ** c
        # i_gamma = 0
        for gamma in xrange(gstart, gend - 1, gstep):
            # print 'IGAMMA', i_gamma
            # i_gamma += 1 
            param.gamma = 2.0 ** gamma # <- Could that be a problem for linear kernel?
            key = (c, gamma)
            # t00 = time.clock()
            grid[key] = grid.get(key, []) + cv_func(y, x, param, n=n)
            # t11 = time.clock()
            # print 'Search took', t11 - t00, 'seconds.'
    # t1 = time.clock()
    # print 'Time:', t1 - t0
    return grid

###
def leave_one_out(y, x, param, n='DUMMY'):
    results = []
    for i, test in enumerate(zip(y, x)):
        training_y = y[:i] + y[i+1:]
        training_x = x[:i] + x[i+1:]
        problem = svm.svm_problem(training_y, training_x)
        # t0 = time.clock()
        model = svmutil.svm_train(problem, param, '-q')
        # t1 = time.clock()
        # print 'Training took', t1 - t0, 'seconds.'
        result = svmutil.svm_predict(y[i:i+1], x[i:i+1], model, '-b 1')
        results.append(result + (test[0], make_d.decode(x[i], make_d.decode_dic)))
    return results

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

    init(argv[2:])    
    print SET1, SET2
    
    param = svm.svm_parameter('-b 1')
    if KERNEL_TYPE == 'LINEAR':
        param.kernel_type = svm.LINEAR
        GAMMA_RANGE = 1, 0, -2
    else:
        param.kernel_type = svm.RBF

    cvfunc = leave_one_out
    n_cv = None

    use_sets = not SET1 is None and not SET2 is None
    
    fn = argv[0]
    dataset = make_d.read_data(open(fn))
    data = make_d.assign_classes(dataset)

    data = [(d[0], d[1][1:]) for d in data]

    data = make_d.prepare_data(data)
    """ Next line is just for testing. """
    data = {1.0: data[1.0], 0.0: data[0.0]}
    print data.keys(), [len(v) for v in data.values()]    
    
    testdata = make_d.read_data(open(argv[1]))    
    testset = make_d.assign_classes(testdata)

    testset = [(d[0], d[1][1:]) for d in testset]

    testset = make_d.prepare_data(testset)

    precursor = {}
    for k, v in testdata.items():
        v = v[1:]
        precursor[v] = precursor.get(v, []) + [int(k.split('_')[-1])]    
    print precursor
    
    outfile = os.path.basename(fn)
    outfile = outfile.replace('.fasta', '')
    outfile = outfile.replace('.fas', '')
    if use_sets:
        outfile = ''.join(map(str, map(int, SET1))) + 'vs' + ''.join(map(str, map(int, SET2)))

    log_name = '%s-%s-%i-%s.csv' % (TIMESTAMP, 
                                    KERNEL_TYPE,
                                    int(RANDOMIZE_DATA),
                                    outfile)
    logfile = open(log_name, 'w')         
    
    """ Prepare test set (precursor fragments). """                
    testset[-1.0] = copy.deepcopy(testset[0.0])
    del testset[0.0]
    testset = make_d.make_set(testset, balanced_set=False, training_fraction=1.0)
    """ 'Training' and 'Test' sets flipped """
    test_y, test_x = testset[:2]
    encoded_x = [make_d.encode(x, make_d.encode_dic) for x in test_x]
    
    # logfile.write(',%s\n' % ','.join(map(str, map(int, test_y))))
    

    """ Train and predict """
    row = [0.0 for x in test_x]
    while i < N_RUNS:
        sys.stdout.write('%i ' % i)
        sys.stdout.flush()
        
        set1 = dict([item for item in data.items()
                     if item[0] == 1.0])
        set2 = dict([item for item in data.items()
                     if item[0] == 0.0])
        set1 = make_d.make_set(set1, training_fraction=1.0)
        set2 = make_d.make_set(set2, training_fraction=1.0)
        new_sets = {1.0: set1[1], -1.0: set2[1]}
        sets = make_d.make_set(new_sets, training_fraction=1.0)
        train_y, train_x, dummy_y, dummy_x = sets
    
        print [len(x) for x in sets]
        train_x = [make_d.encode(x, make_d.encode_dic) for x in train_x]
        
        t0 = time.clock()
        param_grid = {}
        
        param_grid = grid_search(train_y, train_x, param, param_grid,
                                 cvfunc, n_cv, C_RANGE, GAMMA_RANGE)
        t1 = time.clock()
        print 'Time:', t1 - t0, 'Remaining:', (N_RUNS-(i+1)) * (t1 - t0)

        ranking = []
        for k, v in param_grid.items():
            recognized = [v_i[0][0] == v_i[3] for v_i in v]
            recog_rate = sum(map(int, recognized))/float(len(recognized))
            # print k, recog_rate, len(recognized)
            ranking.append((recog_rate, k))
        ranking.sort()
        
        param.C, param.gamma = map(lambda x: 2**x, ranking[-1][1])
        problem = svm.svm_problem(train_y, train_x)
        model = svmutil.svm_train(problem, param, '-q')
        
	result = svmutil.svm_predict(test_y, encoded_x, model, '-b 1')
        ## print result
        ## print zip(test_y, test_x)

        cur_result = zip(result[0], test_y)
        
        for row_i, res in enumerate(result[0]):
            if res == -1.0:
                res = 0.0
            row[row_i] += res

        # logfile.write('%i,%s\n' % (i, ','.join(map(str, map(int, result[0])))))
	i += 1        
        pass

    row =  map(lambda x: x/N_RUNS, row)

    pos_on_precursor = []
    for dat in zip(row, test_x, test_y):
        #print dat
        pos_on_precursor.append((precursor[dat[1]][0],) +  dat)
        del precursor[dat[1]][0]
    for dat in sorted(pos_on_precursor):
        print dat
        logfile.write('%05i,%f,%s,%f\n' % dat)


    logfile.close()
    return None
    
    


if __name__ == '__main__': main(sys.argv[1:])
