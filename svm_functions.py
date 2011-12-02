#!/usr/bin/python

import os
import sys
import math
import time

LIBSVM_PATH = '/home/schudoma/tools/libsvm-3.1/python'
if not os.path.exists(LIBSVM_PATH):
    sys.stderr.write('Missing LIBSVM_PATH. Aborting.\n')
    sys.exit(1)    
sys.path.append(LIBSVM_PATH)
import svm
import svmutil

import make_svm_data as make_d

###
def grid_search(y, x, param, cv_func, n, c_range, gamma_range):
    cstart, cend, cstep = c_range
    gstart, gend, gstep = gamma_range
    grid = {}
    for c in xrange(cstart, cend + 1, cstep):
        param.C = 2.0 ** c
        for gamma in xrange(gstart, gend - 1, gstep):
            """ Could next line be a problem for linear kernel? """
            param.gamma = 2.0 ** gamma 
            key = (c, gamma)
            grid[key] = grid.get(key, []) + cv_func(y, x, param, n=n)
    
    ranking = []
    for k, v in grid.items():
        recognized = [v_i[0][0] == v_i[3] for v_i in v]
        recog_rate = sum(map(int, recognized))/float(len(recognized))
        ranking.append((recog_rate, k))
    ranking.sort()

    print param.C, param.gamma
    param.C, param.gamma = map(lambda x: 2**x, ranking[-1][1])
    return param

###
def leave_one_out(y, x, param, n=None):
    results = []
    for i, test in enumerate(zip(y, x)):
        training_y = y[:i] + y[i+1:]
        training_x = x[:i] + x[i+1:]
        problem = svm.svm_problem(training_y, training_x)
        model = svmutil.svm_train(problem, param, '-q')
        result = svmutil.svm_predict(y[i:i+1], x[i:i+1], model, '-b 1')
        results.append(result + (test[0], 
                                 make_d.decode(x[i], make_d.decode_dic)))
    return results


###
def main(argv):
    return None

if __name__ == '__main__': main(sys.argv[1:])
