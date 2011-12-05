#!/usr/bin/python

import os
import sys
import math
import time
import random

from optparse import OptionParser

import make_svm_data as make_d


###
class MLRunner(object):

    #
    def __init__(self):
        self.timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.n_runs = 1000
        self.set1 = None
        self.set2 = None
        self.randomize_data = False
        self.limit_sets = False

        self._optparser = OptionParser()
        self._optparser.add_option('-n', type='int', dest='n_runs')
        self._optparser.add_option('--s1', dest='set1')
        self._optparser.add_option('--s2', dest='set2')
        self._optparser.add_option('--random', dest='randomize_data', 
                                   action='store_true')

        self.logfile = None
        self.data = None

        pass

    #
    def set_parameters(self, argv):
        options, args = self._optparser.parse_args(argv)
        print options
        print options.__dict__.items()
        for key, val in options.__dict__.items():
            setattr(self, key, val)
        
        if options.set1 is not None and options.set2 is not None:
            self.set1 = [float(v) for v in options.set1.split(':')]
            self.set2 = [float(v) for v in options.set2.split(':')]
            self.limit_sets = True
        self.randomize_data = options.randomize_data is not None

        return options, args

    #
    def start_logging(self, logname_components):
        """ logname_components[-1] must be the input filename! """
        if self.limit_sets:
            inputfile = 'vs'.join([''.join(map(str, map(int, self.set1))),
                                   ''.join(map(str, map(int, self.set2)))])
        else:
            inputfile = os.path.basename(logname_components[-1])
            inputfile = inputfile.strip('ta').strip('.fas')

        logname = '-'.join([self.timestamp] + \
                               logname_components[:-1] + [inputfile])
        self.logfile = open(logname + '.csv', 'w')
        pass

    #
    def write_log(self, string):
        if self.logfile is None:
            pass
        else:
            self.logfile.write(string)
            self.logfile.flush()
        pass
    
    #
    def stop_logging(self):
        if logfile is not None:
            self.logfile.close()
        pass

    #
    @staticmethod
    def compute_accuracy(results):
        f = lambda x: x[0]==x[1]
        return sum(map(float, map(f, results)))/len(results)

    #
    def setup(self, argv):        
        self.set_parameters(argv[1:])
        print self.set1, self.set2
        print self.__dict__.items()
        
        self.data = make_d.read_data(open(argv[0]))
        self.data = make_d.assign_classes(self.data)
        self.data = make_d.prepare_data(self.data)
        print self.data.keys(), [len(v) for v in self.data.values()]
        pass
    
    
    #
    def run(argv):
        pass
    

def main(argv):
    return None


if __name__ == '__main__': main(sys.argv[1:])

