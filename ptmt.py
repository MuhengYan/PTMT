"""
#Author: Muheng Yan
#Git Repo: https://github.com/MuhengYan/
#Licence: GNU General Public License (v3)
"""

"""
Reference: Stanford Topic Modeling Toolbox
Website: http://nlp.stanford.edu/software/tmt/tmt-0.4/
The distributions of Stanford Topic Modeling Toolbox are
open source under GNU General Public License (v3).

Inspired by "topbox" by Chris Emmery "https://github.com/clips/topbox/"
which doesn't support updated environments (simply doesn't work)
and doesn't support extra parameters.
This package re-constructed the framework and added support
for extra costume training parameters of Labeled LDA
"""

from subprocess import call
from os import path, sep


class PythonTMT:
    """
    Bla Bla Bla
    """
    def __init__(self,name,maxIter = 500, filter = [1,1,1,1,1]):
        self.name = name
        self.maxIter = maxIter
        self.filter = filter
        self.dir = path.dirname(path.realpath(__file__)) + \
                   '{0}toolbox{}'.format(sep)

    def callTMT(self, mode, csvfile="something"):
        modelName = self.name
        call([])

    def store(self):
        print("yay")

    def rm(self):
        print("yay")

    def train(self, labels, texts):
        self.store()
        self.callTMT(mode='train')
        self.rm()

    def trainFromCSV(self, file):
        self.callTMT(mode='train', csvfile=file)
        self.rm()

    def infer(self, texts):
        self.store()
        self.callTMT(mode = 'infer')

    def inerFromCSV(self, file):
        self.callTMT(mode='infer', csvfile=file)

    def evaluation(self, trueLabels):
        print("yay")

    def evaluationFromCSV(self, file):
        print("yay")