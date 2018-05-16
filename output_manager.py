import os
from parameters_sampler import ParametersSampler
from shutil import copyfile

import numpy

import time
import datetime


class OutputManager():
    def __init__(self,data_aug=False,filtered=False,name_test=""):
        if not os.path.exists('output'):
            os.makedirs('output')
        self.output_dir = "output/" + name_test + datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S") +"_DA_"+str(data_aug)+"_filtered_"+str(filtered)
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

    def write_one(self, output, count):
        numpy.savetxt(self.output_dir + "/run" + str(count) + ".csv", output, delimiter=",", fmt="%.4f")
