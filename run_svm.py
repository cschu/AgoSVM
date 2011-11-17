#!/usr/bin/python

import os
import sys
import math
import time
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

def init(argv):
    global TIMESTAMP
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())            
    parser = OptionParser()
    parser.add_option('-k', '--kernel', dest='kernel_type')
    parser.add_option('-n', type='int', dest='n_runs')
    parser.add_option('--random', dest='randomize_data', action='store_true')
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

    return None

###
def grid_search(y, x, param, grid, cv_func, n, c_range, gamma_range):
    cstart, cend, cstep = c_range
    gstart, gend, gstep = gamma_range
    for c in xrange(cstart, cend + 1, cstep):
        param.C = 2.0 ** c
        for gamma in xrange(gstart, gend - 1, gstep):
            param.gamma = 2.0 ** gamma
            key = (c, gamma)
            grid[key] = grid.get(key, []) + cv_func(y, x, param, n=n)
    return grid

###
def leave_one_out(y, x, param, n='DUMMY'):
    # print 'XY', zip(y, x)
    results = []
    for i, test in enumerate(zip(y, x)):
        training_y = y[:i] + y[i+1:]
        training_x = x[:i] + x[i+1:]
        problem = svm.svm_problem(training_y, training_x)
        model = svmutil.svm_train(problem, param, '-q')
        result = svmutil.svm_predict(y[i:i+1], x[i:i+1], model, '-b 1')
        results.append(result + (test[0], make_d.decode(x[i], make_d.decode_dic)))
    return results


###
def main(argv):

    global C_RANGE
    global GAMMA_RANGE

    init(argv[1:])    

    fn = argv[0]
    dataset = make_d.read_data(open(fn))

    # print dataset
    # print 'X'
    items = dataset.items()
    keys = [float(x[0].split('_')[0][3]) for x in items]
    dataset = zip(keys, [v[1] for v in items])
    """
    dataset = {}
    for k, v in zip(keys, items):
        dataset[k] = dataset.get(k, []) + [v[1]]
    """
    # print dataset
    
    """
    for k1, k2 in zip(keys, dataset.keys()):
        print k1, k2, k1 == float(k2.split('_')[0][3])
    return None
    """

    data = make_d.prepare_data(dataset)
    print data.keys(), [len(v) for v in data.values()]

    param = svm.svm_parameter('-b 1')
    if KERNEL_TYPE == 'LINEAR':
        param.kernel_type = svm.LINEAR
        GAMMA_RANGE = 1, 0, -2
    else:
        param.kernel_type = svm.RBF

    cvfunc = leave_one_out
    n_cv = None

    i = 0
    param_grid = {}
    results = []
    while i < N_RUNS:
        print i
        sets = make_d.make_set(data, training_fraction=0.75)
        train_y, train_x, test_y, test_x = sets

        print [len(x) for x in sets]
        
        train_x = [make_d.encode(x, make_d.encode_dic) for x in train_x]
        test_x  = [make_d.encode(x, make_d.encode_dic) for x in test_x]
        
        print len(train_x), len(test_x)
        
        param_grid = {}
        param_grid = grid_search(train_y, train_x, param, param_grid,
                                 leave_one_out, n_cv, C_RANGE, GAMMA_RANGE)

        ranking = []
        for k, v in param_grid.items():
            recognized = [v_i[0][0] == v_i[3] for v_i in v]
            recog_rate = sum(map(int, recognized))/float(len(recognized))
            print k, recog_rate, len(recognized)
            ranking.append((recog_rate, k))
        ranking.sort()
    
        param.C, param.gamma = map(lambda x: 2**x, ranking[-1][1])
        problem = svm.svm_problem(train_y, train_x)
        model = svmutil.svm_train(problem, param, '-q')

        result = svmutil.svm_predict(test_y, test_x, model, '-b 1')
        print result
        print test_y
        results.extend(zip(result[0], test_y))

        i += 1
        pass

    print 'ACC', sum(map(float, map(lambda x: x[0]==x[1], results)))/len(results)


    return None
    print param_grid
    print param_grid.keys()
    print
    print param_grid.items()[0]

    ranking = []
    for k, v in param_grid.items():
        recognized = [v_i[0][0] == v_i[3] for v_i in v]
        recog_rate = sum(map(int, recognized))/float(len(recognized))
        print k, recog_rate, len(recognized)
        ranking.append((recog_rate, k))
    ranking.sort()
    
    param.C, param.gamma = map(lambda x: 2**x, ranking[-1][1])
    problem = svm.svm_problem(train_y, train_x)
    model = svmutil.svm_train(problem, param, '-q')

    result = svmutil.svm_predict(test_y, test_x, model, '-b 1')
    print result
    
    return None

if __name__ == '__main__': main(sys.argv[1:])
