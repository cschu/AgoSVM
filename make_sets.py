#!/usr/bin/python

import os
import sys
import math

import make_svm_data as make_d

###
def do_something():
    return None 

###
def write_set(set_, fo):    
    for i, (k, v) in enumerate(set_):
        fo.write('>ago%i_%05i\n%s\n' % (k, i, v))
    return None

###
def main(argv):

    fn = argv[0]
    dataset = make_d.read_data(open(fn))
    negset = make_d.read_data(open(argv[1]))
    
    items = dataset.items()
    keys = [float(x[0].split('_')[0][3]) for x in items]
    dataset = zip(keys, [v[1] for v in items])

    negitems = negset.items()
    negkeys = [float(x[0].split('_')[0][3]) for x in negitems]
    negset = zip(negkeys, [v[1] for v in negitems])               

    set1 = [x for x in dataset if x[0] == 1.0]
    set2 = [x for x in dataset if x[0] == 2.0]
    set4 = [x for x in dataset if x[0] == 4.0]
    set0 = [x for x in negset]
        
    set1_2 = [(1.0, x[1]) for x in set1] + [(-1.0, x[1]) for x in set2]
    write_set(set1_2, open('1vs2.fas', 'w'))
    set1_4 = [(1.0, x[1]) for x in set1] + [(-1.0, x[1]) for x in set4]
    write_set(set1_4, open('1vs4.fas', 'w'))
    set2_4 = [(1.0, x[1]) for x in set2] + [(-1.0, x[1]) for x in set4]
    write_set(set2_4, open('2vs4.fas', 'w'))

    set12_4 = [(1.0, x[1]) for x in set1 + set2] + [(-1.0, x[1]) for x in set4]
    write_set(set12_4, open('12vs4.fas', 'w'))
    set14_2 = [(1.0, x[1]) for x in set1 + set4] + [(-1.0, x[1]) for x in set2]
    write_set(set12_4, open('14vs2.fas', 'w'))
    set24_1 = [(1.0, x[1]) for x in set2 + set4] + [(-1.0, x[1]) for x in set1]
    write_set(set12_4, open('24vs1.fas', 'w'))

    set_all = set1 + set2 + set4 + set0
    write_set(set12_4, open('1vs2vs4vs0.fas', 'w'))
    
    set1_0 = [(1.0, x[1]) for x in set1] + [(-1.0, x[1]) for x in set0]
    write_set(set1_0, open('1vs0.fas', 'w'))
    set2_0 = [(1.0, x[1]) for x in set2] + [(-1.0, x[1]) for x in set0]
    write_set(set2_0, open('2vs0.fas', 'w'))
    set4_0 = [(1.0, x[1]) for x in set4] + [(-1.0, x[1]) for x in set0]
    write_set(set4_0, open('4vs0.fas', 'w'))

    set12_0 = [(1.0, x[1]) for x in set1 + set2] + [(-1.0, x[1]) for x in set0]
    write_set(set12_0, open('12vs0.fas', 'w'))
    set14_0 = [(1.0, x[1]) for x in set1 + set4] + [(-1.0, x[1]) for x in set0]
    write_set(set14_0, open('14vs0.fas', 'w'))
    set24_0 = [(1.0, x[1]) for x in set2 + set4] + [(-1.0, x[1]) for x in set0]
    write_set(set24_0, open('24vs0.fas', 'w'))
    
    

    return None

if __name__ == '__main__': main(sys.argv[1:])
