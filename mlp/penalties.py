import numpy as np

seed = 22102017
rng = np.random.RandomState(seed)


class L1Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """

    def __init__(self, coefficient):
        """Create a new L1 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient * np.sum(np.abs(parameter))
        


    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient * np.sign(parameter)
        

    def __repr__(self):
        return 'L1Penalty({0})'.format(self.coefficient)


class L2Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient * 0.5 * np.sum(parameter ** 2)

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient * parameter

    def __repr__(self):
        return 'L2Penalty({0})'.format(self.coefficient)

class L1L2MixPenalty(object):
    """L1 & L2 mix penalty.
    """

    def __init__(self, lambda_param, alpha):
        """Create a new L1 & L2 mix penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
         
        # 1. Parameter Validation
        assert lambda_param >= 0., 'lambda_param must be non-negative.'
        assert 0. <= alpha <= 1., 'alpha must be between 0 and 1.'

        self.lambda_param = lambda_param
        self.alpha = alpha
        
        # 2. Map standard parameters to your internal coefficients
        # Your l1_coeff (lambda_1) = lambda_param * alpha
        # Your l2_coeff (lambda_2) = lambda_param * (1 - alpha)
        
        # L1 coefficient (controls the sparsity/feature selection)
        self.l1_coeff = lambda_param * alpha

        # L2 coefficient (controls the magnitude/grouping)
        self.l2_coeff = lambda_param * (1.0 - alpha)


    

    def __call__(self, parameter):
        """Calculate L1 & L2 mix penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        
        return (self.l1_coeff * np.sum(np.abs(parameter)) +
                0.5 * self.l2_coeff * np.sum(parameter ** 2))

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        
        # Gradient = l1_coeff * sign(w) + l2_coeff * w
        return self.l1_coeff * np.sign(parameter) + self.l2_coeff * parameter

    def __repr__(self):
           return 'L1L2MixPenalty(lambda_param={0}, alpha={1})'.format(self.lambda_param, self.alpha)
