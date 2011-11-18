#!/usr/bin/python
#
# make_svm_data.py
#
# Functions for preparing data for libSVM experiments.
#
#
#

__author__ = 'Christian Schudoma'
__copyright__ = 'Copyright 2010-2011, Christian Schudoma'
__credits__ = []
__license__ = 'None'
__version__ = '0.1a'
__maintainer__ = 'Christian Schudoma'
__email__ = 'schudoma@mpimp-golm.mpg.de'
__status__ = 'Development'

"""
 make_svm_data (make_d) - 
 Functions for preparing data for libSVM experiments.

  Author:
     Christian Schudoma

 (c) 2010-2011

"""

import os
import sys
import math
import copy
import random

#
decode_dic = {(-1,-1,-1,-1,1): 'A',
              (-1,-1,-1,1,-1): 'C',
              (-1,-1,1,-1,-1): 'G',
              (-1,1,-1,-1,-1): 'U',
              (1,-1,-1,-1,-1): '$'}

#
encode_dic = {'A': [-1,-1,-1,-1,1],
              'C': [-1,-1,-1,1,-1],
              'G': [-1,-1,1,-1,-1],
              'U': [-1,1,-1,-1,-1],
              '$': [1,-1,-1,-1,-1]}

#
def draw_n_numbers(n, urn):
    """
    *draw_n_numbers(n, urn)*

    Randomly draws n integers from an urn. The urn is emptied in the process.

    Arguments:
       * n - the number of integers to be drawn
       * urn - the list of integers to be drawn from

    Returns:
       * list of integers
    """

    numbers = []
    while True:
        if len(urn) == 0 or len(numbers) == n: break
        p = random.randint(0, len(urn) - 1)
        numbers.append(urn[p])
        del urn[p]
    return numbers


#
def n_partition_set(n, setsize):
    """
    *n_partition_set(n, setsize)*

    Computes random n-partitions for sets of a given size.

    Arguments:
       * n - the size of the partitions
       * setsize - the size of the set

    Returns:
       * list of n-partitions (bins)
    """

    p_size = setsize / n
    urn = range(setsize)
    bins = []
    for i in xrange(n):
        bins.append(draw_n_numbers(p_size, urn))
    i = 0
    while True:
        if len(urn) == 0: break
        bins[i] += draw_n_numbers(1, urn)
        i += 1
        if i >= len(bins): i = 0
        pass
    return bins

#
def read_data(fi):
    """
    *read_data(fi)*

    Reads data from an unformatted fasta file.
    REQUIRES SEQUENCES WITH UNIQUE IDENTIFIERS!

    Arguments:
       * a file handle

    Returns:
       * a dictionary {seqid: seqdata}
    """

    data = {}
    last = None
    while True:
        line = fi.readline()
        if not line: break
        if line[0] == '>':
            last = line[1:].strip()
        else:
            data[last] = line.strip()
        pass
    
    return data

#
def prepare_data(data):
    """ Identify instances that are shared between classes. """
    shared = {}
    for k, v in sorted(data):
        class_ = float(k)
        shared[v] = shared.get(v, set()).union(set([k]))
    # print shared
    """ Remove duplicate and shared instances. """
    unique_data = {}
    for v, classes in shared.items():
        if len(classes) == 1:
            k = list(classes)[0]
            unique_data[k] = unique_data.get(k, []) + [v]

    return unique_data
        
#
def n_random_sequences(n, length=10):
    sequences = set()
    while True:
        if len(sequences) == n: break
        seq = ['ACGU'[random.randint(0, 3)] for i in xrange(length)]
        sequences.add(''.join(seq))
    return list(sequences)

#
def decode(string, dic):
    """
    *decode(string, dic)*

    Decodes a binary vector using a dictionary.

    Arguments:
       * string - a binary vector
       * a dictionary {vector: char}

    Returns:
       * the decoded string
    """

    decoded = ''
    i = 0
    step = len(dic.keys()[0])
    while i < len(string):
        decoded += dic[tuple(string[i:i+step])]
        i += step
    return decoded
    
#
def encode(string, dic, maxlen=None):
    """
    *encode(string, dic, maxlen)*

    Encodes a string as a binary vector, 
    padding shorter sequences at the 3'-end with '$'-guards.

    Arguments:
       * a string
       * a dictionary {char: vector}
       * the maxlen used to determine the number of 3'-$-guards

    Returns:
       * the encoded string
    """
    encoded = []
    for c in string:
        encoded.extend(dic[c])
    if not maxlen is None:
        for i in xrange(maxlen - len(string)):
            encoded.extend(dic['$'])
    return encoded
    
#
def balance_data(data, minsize):     
    """
    *balance_data(data, minsize)*

    Balances a data set with subsets of unequal sizes.
    
    Arguments:
       * a data dictionary {class: [sequences]}
       * minsize - the size of the smallest class

    Returns:
       * the balanced data dictionary {class: [sequences]}
    """

    balanced = {}
    for key, val in copy.deepcopy(data.items()):
        if len(val) == minsize:
            balanced[key] = [v for v in val]
        else:
            numbers = draw_n_numbers(minsize, range(len(val)))
            balanced[key] = [val[ix] for ix in numbers]
        pass
    return balanced

#
def make_set(data, balanced_set=True, training_fraction=0.5):
    """
    *make_set(data, balanced_set=True, training_fraction=0.5)*

    Creates test and training sets from a data set.

    Arguments:
       * a data dictionary {class: [sequences]}
       * a balancing flag
       * a parameter determining the size of the training set

    Returns:
       * 4 lists: training labels/features, test labels/features
    """

    minsize = min([len(val) for key, val in data.items()])
    if balanced_set:
        dataset = balance_data(data, minsize)
    else:
        dataset = copy.deepcopy(data)
        pass   

    training_y, training_x = [], []
    test_y, test_x = [], []
    
    for k, val in dataset.items():
        training_size = int(math.ceil(len(val) * training_fraction))
        training_ = draw_n_numbers(training_size, range(len(val)))
        for i, v in enumerate(val):
            if i in training_:
                training_y.append(k)
                training_x.append(v)
            else:
                test_y.append(k)
                test_x.append(v)
        pass
    
    return training_y, training_x, test_y, test_x

#
def write_set(y, x, fo):
    """
    *write_set(y, x, fo)*
    
    Writes a data set (labels, features) to a file.

    Arguments:
       * set labels
       * set features
       * a (writing) file handle
    """
    
    for yx in zip(y, x):
        row = ['%i:%i' % (i, int(xx)) for i, xx in enumerate(yx[1])]
        fo.write('%s\n' % ' '.join([str(y)] + row))
    return None
            
  



###
def main(argv):

    data = read_data(open(argv[0]))
    data = make_set(data)

    for k, v in sorted(data.items()):
        print k, len(v)
        for seq in sorted(list(v)):
            print seq

    return None

if __name__ == '__main__': main(sys.argv[1:])


