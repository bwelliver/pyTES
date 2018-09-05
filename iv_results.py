class FitParameters:
    
    def __init__(self):
        self.left = FitResult()
        self.right = FitResult()
        self.sc = FitResult()
    def __repr__(self):
        """return string representation."""
        s = 'Left Branch:\n' + str(self.left)
        s = s + '\n' + 'Right Branch:\n' + str(self.right)
        s = s + '\n' + 'Superconducting:\n' + str(self.sc)
        return(s)


class FitResult:
    
    def __init__(self):
        self.result = None
        self.error = None
    def set_values(self, result=None, error=None):
        self.result = result
        self.error = error
    def __repr__(self):
        """return string representation."""
        s = '\t' + 'Result:\t' + str(self.result)
        s = s + '\n' + '\t' + 'Error:\t' + str(self.error)
        return(s)
