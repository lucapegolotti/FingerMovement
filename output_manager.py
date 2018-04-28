import os
from parameters_sampler import ParametersSampler
from shutil import copyfile


import time
import datetime


class OutputManager():
    def __init__(self):
        if not os.path.exists('output'):
            os.makedirs('output')

    def write(self, train_error_string, test_error_string, parameters_sampler):
        nameFile = "output/out" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S") + ".txt";
        outputFile = open(nameFile,"w")

        outputFile.write(train_error_string + "\n")
        outputFile.write(test_error_string + "\n\n")
        outputFile.write(parameters_sampler.getDescriptions() + "\n")

        outputFile.close()

        # swap with best if the result is better
        mustCopy = True
        for filename in os.listdir("output"):
            if filename.startswith("best"):
                f = open("output/" + filename, "r")
                f.readline()
                test_error = self.retrieveTestErrorFromString(f.readline())
                curr_test_error = self.retrieveTestErrorFromString(test_error_string)

                if (float(curr_test_error) > float(test_error)):
                    mustCopy = False

        if (mustCopy):
            copyfile(nameFile, "output/best.txt")

    def retrieveTrainErrorFromString(self, train_error):
        out = train_error[13:]
        return out[:-2]

    def retrieveTestErrorFromString(self, train_error):
        out = train_error[12:]
        return out[:-2]
