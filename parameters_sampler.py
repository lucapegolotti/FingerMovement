import random
from abc import ABC, abstractmethod
import numpy as np

class Parameter(ABC):
    def __init__(self, type, size):
        super().__init__()

        if (type != "int" and type != "float"):
            raise ValueError("parameter value type not supported")
        self.type = type
        self.size = size

    @abstractmethod
    def doSample(self):
        pass

    def sample(self):
        self.value = self.doSample()
        if (self.type == "int"):
            self.value = self.value.astype(int)

    def valueToString(self):
        out = ""
        for i in range(0, self.size):
            out += str(self.value[i])
            if (i != self.size - 1):
                out += ","
        return out

    def myDescription(self):
        myDescription = "";
        myDescription += "Type = " + self.type + "\n"
        myDescription += "Size = " + str(self.size) + "\n"
        myDescription += "Value = " + self.valueToString() + "\n"
        return myDescription

    def showMe(self):
        print(self.myDescription())

class UniformParameter(Parameter):
    def __init__(self, type, size, minValue, maxValue):
        super().__init__(type, size)
        self.minValue = minValue
        self.maxValue = maxValue
        self.sample()

    def doSample(self):
        return np.random.uniform(self.minValue, self.maxValue, size = self.size)

class NormalParameter(Parameter):
    def __init__(self, type, size, meanValue, stdValue):
        super().__init__(type, size)
        self.meanValue = meanValue
        self.stdValue = stdValue
        self.sample()

    def doSample(self):
        return np.random.normal(self.meanValue, self.stdValue, size = self.size)


class ParametersSampler():
    def __init__(self):
        self.parameters = {'batch_size': UniformParameter("int", 1, 40, 80),
                           'eta': NormalParameter("float", 1, 0.15, 0.05),
                           'epochs': UniformParameter("int", 1, 100, 1000),
                           'dropout': NormalParameter("float", 1, 0.1, 0.05)}
    def showMe(self):
        print("List of parameters:")
        print("===================")
        for k, v in self.parameters.items():
            print("Parameter = " + k)
            v.showMe()

    def getDescriptions(self):
        descriptions = "List of parameters:\n===================\n"
        for k, v in self.parameters.items():
            descriptions += "Parameter = " + k + "\n"
            descriptions += v.myDescription()
            descriptions += "\n"

        return descriptions

    def getParameter(self, key):
        param = self.parameters[key]
        if (param.size == 1):
            if (param.type == "int"):
                return  int(param.value[0])
            else:
                return  float(param.value[0])
        return param.value
