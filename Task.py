
class Task:

    def __init__(self, alpha, beta, gamma, n):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.h = 1/n

    def u(self, x):
        return x**self.alpha * (1-x)**self.beta
    
    def p(self, x):
        return 1 + x**self.gamma

    def g(self, x):
        return x + 1

    def f(self, x):
        return -2 + 6*x + x**2 - 8*x**3 + 14*x**4

    def a(self, i):
        return self.p(i*self.h)

    def b(self, i=-1):
        if (i==-1):
            return [self.b(i) for i in range(1, self.n)]
        return self.h**2 * self.f(i*self.h)

    def matrix(self, i=-1, j=-1):
        if (i == -1 and j == -1):
            matrix = []
            for i in range(1, self.n):
                row = []
                for j in range(1, self.n):
                    row.append(self.matrix(i, j))
                matrix.append(row)
            return matrix

        if (i == 0 or i == self.n):
            return 1 if i==j else 0
        
        # right
        if (i-j == -1):
            if (i == self.n-1):
                return 0
            return -self.a(i)

        # center
        if (i-j == 0):
            return self.a(i) + self.a(i+1) + self.h**2*self.g(i*self.h)

        # left
        if (i-j == 1):
            if (i == 1):
                return 0
            return -self.a(i+1)

        return 0
