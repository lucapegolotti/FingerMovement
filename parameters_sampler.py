import random
from abc import ABC, abstractmethod
import numpy as np
import math

class Parameter(ABC):
    """
    Constructor
    ---------
    Input parameters:
    - type: type of the parameter (e.g. 'int' or 'float')
    - size: size of the array
    """
    def __init__(self, type, size):
        super().__init__()

        if (type != "int" and type != "float"):
            raise ValueError("parameter value type not supported")
        self.type = type
        self.size = size

    """
    Do sample (to be overriden by derived classes): draws parameter from a
    given distribution
    """
    @abstractmethod
    def doSample(self):
        pass

    """
    Sample: Callback to doSample
    """
    def sample(self):
        self.value = self.doSample()
        if (self.type == "int"):
            self.value = self.value.astype(int)

    """
    Value to string: convert numerical value to string
    """
    def valueToString(self):
        out = ""
        for i in range(0, self.size):
            out += str(self.value[i])
            if (i != self.size - 1):
                out += ","
        return out

    """
    My Description: returns string with description of the parameter
    """
    def myDescription(self):
        myDescription = "";
        myDescription += "Type = " + self.type + "\n"
        myDescription += "Size = " + str(self.size) + "\n"
        myDescription += "Value = " + self.valueToString() + "\n"
        return myDescription

    """
    Show me: prints description of the parameter
    """
    def showMe(self):
        print(self.myDescription())

# Parameter drawn from a uniform distribution
class UniformParameter(Parameter):
    """
    Constructor
    ---------
    Input parameters:
    - type: type of the parameter (e.g. 'int' or 'float')
    - size: size of the array
    - minValue: lower bound of the uniform distribution
    - maxValue: upper bound of the uniform distribution
    """
    def __init__(self, type, size, minValue, maxValue):
        super().__init__(type, size)
        self.minValue = minValue
        self.maxValue = maxValue
        self.sample()

    """
    Do sample: sample from a uniform distribution

    Output parameters:
    - vector drawn from uniform distribution
    """
    def doSample(self):
        return np.random.uniform(self.minValue, self.maxValue, size = self.size)

# Odd parameter of the form 2n + 1, where n is dran from a uniform distribution
class UniformOddParameter(Parameter):
    def __init__(self, type, size, minValue, maxValue):
        super().__init__(type, size)
        if minValue % 2 is not 1:
            raise ValueError("Min value of odd parameter must be odd")

        if maxValue % 2 is not 1:
            raise ValueError("Min value of odd parameter must be odd")

        # if the output value is 2n + 1, here we compute an upper and lower bound
        # for n
        self.minValue = (minValue - 1)/2
        self.maxValue = (maxValue - 1)/2
        self.sample()

    """
    Do sample: sample the odd parameter

    Output parameters:
    - a vector with random odd values
    """
    def doSample(self):
        return 2 * np.random.uniform(self.minValue, self.maxValue, size = self.size).astype(int) + 1

# Random variable of the form exp(n), where n is drawn from a uniform distribution
class UniformExponentialParameter(Parameter):
    """
    Constructor
    ---------
    Input parameters:
    - type: type of the parameter (e.g. 'int' or 'float')
    - size: size of the array
    - minExponent: lower bound for the exponent
    - maxExponent: upper bound for the exponent
    """
    def __init__(self, type, base, minExponent, maxExponent):
        super().__init__(type, 1)
        self.minExponent = minExponent
        self.maxExponent = maxExponent
        self.base = base
        self.sample()

    """
    Do sample: sample the exponential parameter

    Output parameters:
    - a vector with values of the form exp(n), with n being drawn from a uniform
      distribution
    """
    def doSample(self):
        return np.array([math.pow(self.base,np.random.uniform(self.minExponent, self.maxExponent))])

# Parameter drawn from a normal distribution
class NormalParameter(Parameter):
    """
    Constructor
    ---------
    Input parameters:
    - type: type of the parameter (e.g. 'int' or 'float')
    - size: size of the array
    - meanValue: value of the mean of the distribution
    - stdValue: value of the standard deviation of the distribution
    """
    def __init__(self, type, size, meanValue, stdValue):
        super().__init__(type, size)
        self.meanValue = meanValue
        self.stdValue = stdValue
        self.sample()

    """
    Do sample: sample the normal parameter

    Output parameters:
    - a vector with normal values
    """
    def doSample(self):
        return np.random.normal(self.meanValue, self.stdValue, size = self.size)

# Manager of multiple parameters
class ParametersSampler():
    """
    Constructor
    """
    def __init__(self):
        # dictionary with all the variables
        self.parameters = {'batch_perc': UniformParameter("float", 1, 0.1, 0.5),
                           'eta': UniformExponentialParameter("float", 10, -5, -2),
                           'size_conv1': UniformParameter("int",1,10,80),
                           'size_conv2': UniformParameter("int",1,10,80),
                           'size_kernel': UniformOddParameter("int",1,3,13),
                           'size_hidden_layer': UniformParameter("int",1,100,150),
                           'scale': UniformParameter("float",1,0.7,0.95),
                           'dropout': UniformParameter("float",1,0.2,0.5),
                           'l2_parameter': UniformExponentialParameter("float", 10, -6, -1),
                            }
    """
    Show me: print description of all the parameters
    """
    def showMe(self):
        print("List of parameters:")
        print("===================")
        for k, v in self.parameters.items():
            print("Parameter = " + k)
            v.showMe()

    """
    Get description: returns string with description of all the parameters
    ---------
    Output parameters:
    - descriptions: string with all the descriptions of the parameters
    """
    def getDescriptions(self):
        descriptions = "List of parameters:\n===================\n"
        for k, v in self.parameters.items():
            descriptions += "Parameter = " + k + "\n"
            descriptions += v.myDescription()
            descriptions += "\n"

        return descriptions

    """
    Get parameter: getter for the value of a particular parameter
    ---------
    Input parameters:
    - key: key of the parameter

    Output parameters:
    - value of the parameter
    """
    def getParameter(self, key):
        param = self.parameters[key]
        if (param.size == 1):
            if (param.type == "int"):
                return  int(param.value[0])
            else:
                return  float(param.value[0])
        return param.value
