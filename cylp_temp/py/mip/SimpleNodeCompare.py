from cylp.py.mip import NodeCompareBase

class SimpleNodeCompare(NodeCompareBase):

    def __init__(self):
        self.method = 'depth'  # For depth-First

    def compare(self, x, y):
        if x.numberUnsatisfied > y.numberUnsatisfied:
            return True
        elif x.numberUnsatisfied < y.numberUnsatisfied:
            return False
        if x.depth != y.depth:
            if self.method == 'depth':
                return x.depth < y.depth
            elif self.method == 'breadth':
                return x.depth > y.depth
        return x.breakTie(y)  # Breaking ties consistently

    def newSolution(self, model,
                            objectiveAtContinuous,
                            numberInfeasibilitiesAtContinuous):
        ''' This method is called after a solution is found
        in a node.
        '''
        self.method = 'breadth'

    def every1000Nodes(self, model, numberNodes):
        ''' This method is called each 1000 nodes for possible
        Changes in strategy
        '''
        return False
