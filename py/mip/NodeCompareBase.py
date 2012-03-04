class NodeCompareBase:
    '''
    Abstract base class
    '''
    def compare(self, x, y):
        '''
        **Abstract Method**

        Return True if ``y`` is better than ``x``
        :arg x: A B&B node
        :type x: CyCbcNode
        :arg y: A B&B node
        :type y: CyCbcNode
        :rtype: bool
        '''
        raise NotImplementedError('SimpleNodeCompare object must ' \
                        'implement *compare*')

    def newSolution(self, model,
                            objectiveAtContinuous,
                            numberInfeasibilitiesAtContinuous):
        '''
        **Abstract Method**

        This method is called after a solution is found
        in a node.
        :rtype: None
        '''
        raise NotImplementedError('SimpleNodeCompare object must ' \
                        'implement *newSolution*')

    def every1000Nodes(self, model, numberNodes):
        '''
        **Abstract Method**

        This method is called each 1000 nodes for possible
        Changes in strategy. Return True to ask for a tree re-sort.
        :rtype: bool
        '''
        raise NotImplementedError('SimpleNodeCompare object must ' \
                        'implement *every1000Nodes*')
