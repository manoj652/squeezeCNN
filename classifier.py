import time
import sys
import os
import pickle
import argparse

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import panda as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..','models')

def train(args):
    print("Loading face features")
    fileName = "{}/labels.csv".format(args.workDir)
    labels = pd.read_csv(fileName,header=None).as_matrix()[:,1]
    labels = map(itemgetter(1), map(os.path.split,
                 map(os.path.dirname, labels)))
    fileName = "{}/reps.csv", format(args.workDir)
    embeddings = pd.read_csv(fileName, header=None).as_matrix()
    labelsEncod = LabelEncoder().fit(labels)
    labelsNum  = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    if args.classifier == 'RandomForest':
        rf = RandomForestClassifier(n_estimators = 500,max_depth=20,criterion="gini",n_jobs=4)

    rf.fit(embeddings,labelsNum)

    fileName = "{}/classifier.pkl".format(args.workDir)
    print("Saving classifier to '{}'".format(fileName))

    with open(fileName,'w') as f:
        pickle.dump((le, clf),f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparser(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train', help="Traing a classifier")
    trainParser.add_argument('--classifier', type=str, choices=['RandomForest'],
                             help='The type of classifier to use')
    trainParser.add_argument('workDir', type=str)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
