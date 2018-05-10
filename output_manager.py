import os
from parameters_sampler import ParametersSampler
from shutil import copyfile

import numpy

import time
import datetime


class OutputManager():
    def __init__(self):
        if not os.path.exists('output'):
            os.makedirs('output')
        self.output_dir = "output/out" + datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        os.makedirs(self.output_dir)

    def write(self, parameters_sampler, outputs):
        nameFile = self.output_dir + "/parameters.txt";
        outputFile = open(nameFile,"w")
        outputFile.write(parameters_sampler.getDescriptions() + "\n")
        outputFile.close()

        count = 0
        for o in outputs:
            count = count + 1
            numpy.savetxt(self.output_dir + "/run" + str(count) + ".csv", o, delimiter=",", fmt="%.4f")

        # swap with best if the result is better
        # mustCopy = True
        # for filename in os.listdir("output"):
        #     if filename.startswith("best"):
        #         f = open("output/" + filename, "r")
        #         f.readline()
        #         test_error = self.retrieveTestErrorFromString(f.readline())
        #         curr_test_error = self.retrieveTestErrorFromString(test_error_string)
        #
        #         if (float(curr_test_error) > float(test_error)):
        #             mustCopy = False
        #
        # if (mustCopy):
        #     copyfile(nameFile, "output/best.txt")

    # def retrieveTrainErrorFromString(self, train_error):
    #     out = train_error[13:]
    #     return out[:-2]
    #
    # def retrieveTestErrorFromString(self, train_error):
    #     out = train_error[12:]
    #     return out[:-2]
