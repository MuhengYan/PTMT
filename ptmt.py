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
from sklearn.metrics import average_precision_score

class PythonTMT:
    """
    reserved for notes
    """
    def __init__(self, name, maxiter=500, filter_llda=[1, 1, 1, 1, 1]):
        """
        reserved for notes
        :param name: str
        the name of the model to be trained
        :param maxiter: int
        the max iteration to train the model
        :param filter_llda: list of int
        label, term and doc filters
        """
        self.name = name
        self.maxiter = maxiter
        self.filter_llda = filter_llda
        self.dir = path.dirname(path.realpath(__file__)) + \
            '{0}toolbox{0}'.format(sep)
        self.modelDir = path.dirname(path.realpath(__file__)) + \
            '{0}'.format(sep)

    def callTMT(self, mode):
        """
        reserved for notes
        :param mode: str
        'train' to call training scala, 'infer' to call inference scala
        :return: None
        """
        if mode == 'train':
            call(["java", "-jar", self.dir + "tmt-0.4.0.jar", self.dir + "llda_" + mode + ".scala",
                  str(self.maxiter), str(self.filter_llda[0]), str(self.filter_llda[1]),
                  str(self.filter_llda[2]), str(self.filter_llda[3]), str(self.filter_llda[4]),
                  self.name, self.dir])
        elif mode == 'infer':
            call(["java", "-jar", self.dir + "tmt-0.4.0.jar", self.dir + "llda_" + mode + ".scala",
                  self.name, self.dir])

    def rm(self, target = 'gz'):
        """
        reserved for notes
        :param target: str
        the target file(s) ti remove, default '.gz' files
        :return: None
        """
        if target == 'gz':
            filelist = glob(dir + "*.gz")
        elif target == 'csv':
            filelist = glob(dir + "*.csv")
        ###\elif target == 'model'
        for file in filelist:
            remove(file)

    def train(self, labels, texts):
        """
        reserved for notes
        :param labels: list of str
        the list of labels of each training sample
        :param texts: list of str
        the list of texts of each training sample
        :return: None
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
        :return: None
        """
        self.callTMT(mode='train')
        self.rm()

    def infer(self, texts):
        """
        reserved for notes
        :param texts: list of str
        the list of text materials you want the existing model to inference
        :return: None
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
        """
        calculate the precision
        use combined with 'infer' or 'inferFromCSV'
        :param trueLabels: list of str
        the true labels of text materials of performed inference
        :return: None
        """
        labelname = open("%s%s%s%s.txt" %
                         (self.modelDir, self.name, sep,
                          '00000{0}label-index'.format(sep)), 'r')
        label_index = labelname.read().lower().split('\n')[:-1]
        labelname.close()

        DTdistro = open("%s%s%s%s.csv" % (self.modelDir, self.name, sep, 'infer-document-topic-distributions-res'))
        predicted_weights = reader(DTdistro)

        ytrue = []
        yprob = []
        for predicted_row, true_row in zip(predicted_weights, trueLabels):
            vector = [(1 if label_index[i] in true_row.lower().split() else 0, float(predicted_row[i + 1]))
                      for i in range(len(label_index))]
            true, prob = zip(*vector)
            if 1 in true:
                ytrue.append(true)
                yprob.append(prob)
        prec = average_precision_score(y_score=yprob, y_true=ytrue)
        return prec