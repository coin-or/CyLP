class cylpCutGenerator:
    '''
    Abstract base class
    '''
    def generateCuts(self, x, y, z):
        '''
        **Abstract Method**

        Return True if ``y`` is better than ``x``
        :arg y: A B&B node
        :type y: CyCbcNode
        :rtype: list
        '''
        raise NotImplementedError('cylpCutGenerator object must ' \
                        'implement *generateCuts*')

