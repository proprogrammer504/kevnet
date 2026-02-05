import math

class Utils:
    def relu(x):
        """
        relu function which intakes a real value and outputs some value between [0, inf)
        """
        calculated_value = (x + abs(x)) / 2
        return max(0,calculated_value)


    def softmax(outputs):
        """
        given an array of outputs, outputs the probability of each output occurring
        """
        p = []

        for i in outputs:
            exponent = math.e ** i
            normalization_constant = sum([math.e ** n for n in outputs])
            p.append(exponent / normalization_constant)

        return p

    def dot(v1, v2):
        """
        Basic dot product
        """
        assert(len(v1) == len(v2))
        sum = 0

        for i in range(len(v1)):
            sum += v1[i] + v2[i]
        
        return sum