import os
import numpy
import time
import datetime
import torch

from parameters_sampler import ParametersSampler
from shutil import copyfile

class OutputManager():

    """
    Constructor: creates the "output" directory (if it does exist) and a directory
    inside output/ with a specific name
    ---------
    Input parameters:
    - data_aug: if True, append DA_True to the folder name
    - filtered: if True, append filtered_True to the folder name
    - name_test: to specify a particular name for the test
    """
    def __init__(self,data_aug=False,filtered=False,name_test=""):
        # create output directory
        if not os.path.exists('output'):
            os.makedirs('output')

        self.output_dir = "output/" + name_test + \
            datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S") + \
            "_DA_"+str(data_aug)+"_filtered_"+str(filtered)

        os.makedirs(self.output_dir)

    """
    Write: write inside the output directory data the parameters and
    the convergence of the loss and errors
    ---------
    Input parameters:
    - parameters_sampler: a ParameterSampler object
    - outputs: list of numpy arrays that will be dumped to file
    """
    def write(self, parameters_sampler, outputs):
        nameFile = self.output_dir + "/parameters.txt";
        outputFile = open(nameFile,"w")
        outputFile.write(parameters_sampler.getDescriptions() + "\n")
        outputFile.close()

        count = 0
        for o in outputs:
            count = count + 1
            numpy.savetxt(self.output_dir + "/run" + str(count) + ".csv", o, delimiter=",", fmt="%.4f")

    """
    Write one: write a single output to file
    ---------
    Input parameters:
    - output: numpy array
    - count: run number (to be appended to file name)
    """
    def writeOne(self, output, count):
        numpy.savetxt(self.output_dir + "/run" + str(count) + ".csv", output, delimiter=",", fmt="%.4f")


    """
    Write model: write a model to a file
    ---------
    Input parameters:
    - model: model to be dumped
    - name: name of the file (without extension)
    """
    def writeModel(self,model,name):
        torch.save(model.state_dict(), self.output_dir + "/" + name + ".pt")
