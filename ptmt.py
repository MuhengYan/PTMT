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
from os import path, sep, remove
from csv import writer, reader
from glob import glob

class PythonTMT:
    """
    reserved for notes
    """
    def __init__(self, name, maxiter=500, filter_llda=[1, 1, 1, 1, 1]):
        """
        reserved for notes
        :param name:
        :param maxiter:
        :param filter_llda:
        """
        self.name = name
        self.maxiter = maxiter
        self.filter_llda = filter_llda
        self.dir = path.dirname(path.realpath(__file__)) + \
                   '{0}toolbox{0}'.format(sep)

    def callTMT(self, mode):
        """
        reserved for notes
        :param mode:
        :return:
        """
        if mode == 'train':
            call(["java", "-jar", dir + "tmt-0.4.0.jar", "llda_" + dir + mode + ".scala",
                  str(self.maxiter), str(self.filter_llda[0]), str(self.filter_llda[1]),
                  str(self.filter_llda[2]), str(self.filter_llda[3]), str(self.filter_llda[4]),
                  self.name, self.dir])
        elif mode == 'infer':
            call(["java", "-jar", dir + "tmt-0.4.0.jar", "llda_" + dir + mode + ".scala",
                  self.name, self.dir])


    def rm(self, target = 'gz'):
        """
        reserved for notes
        :param target:
        :return:
        """
        if target == 'gz':
            filelist = glob(dir + "*.gz")
        elif target == 'csv':
            filelist = glob(dir + "*.csv")

        for file in filelist:
            remove(file)

    def train(self, labels, texts):
        """
        reserved for notes
        :param labels:
        :param texts:
        """
        csv_file = open("%strain.csv" % self.dir, 'w+')
        csv_writer = writer(csv_file)
        for i, zipped in enumerate(zip(labels, texts)):
            line = [str(i + 1), zipped[0], zipped[1]]
            csv_writer.writerow(line)
        csv_file.close()
        #call scala
        self.callTMT(mode='train')
        self.rm()

    def trainFromCSV(self):
        """
        make sure your training data is under ".../PTMT/toolbox"
        the name of your .csv file SHOULD be "train.csv"
        the format of your data SHOULD follow "Serial, Label, Text"
        the .csv file should NOT contain a header
        :return:
        """
        self.callTMT(mode='train')
        self.rm()

    def infer(self, texts):
        """
        reserved for notes
        :param texts:
        :return:
        """
        #save input as .csv file
        csv_file = open("%sinfer.csv" % self.dir, 'w+')
        csv_writer = writer(csv_file)
        for i, zipped in enumerate(zip(texts)):
            line = [str(i + 1), zipped[0]]
            csv_writer.writerow(line)
        csv_file.close()
        # call scala
        self.callTMT(mode='infer')

    def inerFromCSV(self):
        """
        make sure your inference data is under ".../PTMT/toolbox"
        the name of your .csv file SHOULD be "infer.csv"
        the format of your data SHOULD follow "Serial, Text"
        the .csv file should NOT contain a header
        :return:
        """
        self.callTMT(mode='infer')

    def evaluation(self, trueLabels):
        print("yay")

    def evaluationFromCSV(self, file):
        print("yay")