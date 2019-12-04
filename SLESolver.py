import numpy as np


class SLESolver:

    def __init__(self, task):
        super().__init__()
        self.task = task

    def r(self, approx):
        r = []
        for i in range(1, self.task.n):
            r.append(   
                    abs(
                        self.task.matrix(i, i-1)*approx[i-1] +
                        self.task.matrix(i, i)*approx[i] +
                        self.task.matrix(i, i+1)*approx[i+1] -
                        self.task.b(i)
                        )
                    )
        return r
    
    def Jacobi(self, approx=-1, eps=-1):
        result = { 'k': 0, 'approx': []}

        if (approx==-1):
            result['approx'] = [0]*(self.task.n+1)
        else:
            result['approx'] = approx
        if (eps==-1):
            eps = (1/self.task.n)**3

        while True:
            # Невязка
            r = self.r(result['approx'])
            if (max(r) <= eps):
                break

            result['k'] += 1

            next_approx = [0]

            for i in range(1, self.task.n):
                sum1 = 0
                for j in range(1, i):
                    sum1 += result['approx'][j]*self.task.matrix(i, j)/self.task.matrix(i, i)
                sum1 *= -1

                sum2 = 0
                for j in range(i+1, self.task.n):
                    sum2 += result['approx'][j]*self.task.matrix(i, j)/self.task.matrix(i, i)
                sum2 *= -1

                sum3 = self.task.b(i)/self.task.matrix(i, i)

                next_approx.append(sum1 + sum2 + sum3)

            next_approx.append(0)

            result['approx'] = next_approx

        result['approx'] = result['approx'][1:-1]
        return result

    def Seidel(self, approx=-1, eps=-1):
        result = { 'k': 0, 'approx': []}

        if (approx==-1):
            result['approx'] = [0]*(self.task.n+1)
        else:
            result['approx'] = approx
        if (eps==-1):
            eps = (1/self.task.n)**3

        while True:
            # Невязка
            r = self.r(result['approx'])
            if (max(r) <= eps):
                break

            result['k'] += 1

            next_approx = [0]

            for i in range(1, self.task.n):
                sum1 = 0
                for j in range(1, i):
                    sum1 += next_approx[j]*self.task.matrix(i, j)/self.task.matrix(i, i)
                sum1 *= -1

                sum2 = 0
                for j in range(i+1, self.task.n):
                    sum2 += result['approx'][j]*self.task.matrix(i, j)/self.task.matrix(i, i)
                sum2 *= -1

                sum3 = self.task.b(i)/self.task.matrix(i, i)

                next_approx.append(sum1 + sum2 + sum3)

            next_approx.append(0)

            result['approx'] = next_approx

        result['approx'] = result['approx'][1:-1]
        return result

    def __optimizedRelax(self, approx, eps):
        segment = [1, 2]
        optimizedP = 0
        while True:
            # print('segment:({},{})'.format(segment[0], segment[1]))
            tabulation = np.linspace(segment[0], segment[1], 11)
            minK = -1
            minP = -1
            for p in tabulation[1:-1]:
                solution = self.Relax(approx, eps, param=p)
                if minK == -1 or minK > solution['k']:
                    minK = solution['k']
                    minP = p
            minIndex = np.where(tabulation == minP)[0][0]
            segment[0] = tabulation[minIndex-1]
            segment[1] = tabulation[minIndex+1]
            # print('p:{} k:{}'.format(minP, minK))
            if (abs(optimizedP - minP) < 0.001):
                optimizedP = minP
                break
            optimizedP = minP

        solution['p'] = optimizedP
        return solution

    def Relax(self, approx=-1, eps=-1, param=-1, optimize=False):
        result = { 'k': 0, 'approx': []}

        if (approx==-1):
            result['approx'] = [0]*(self.task.n+1)
        else:
            result['approx'] = approx
        if (eps==-1):
            eps = (1/self.task.n)**3
        if (param==-1):
            param = 1.5
        
        if optimize:
            return self.__optimizedRelax(result['approx'], eps)

        while True:
            # Невязка
            r = self.r(result['approx'])
            if (max(r) <= eps):
                break

            result['k'] += 1

            next_approx = [0]

            for i in range(1, self.task.n):
                sum0 = (1 - param)*result['approx'][i]

                sum1 = 0
                for j in range(1, i):
                    sum1 += next_approx[j]*self.task.matrix(i, j)/self.task.matrix(i, i)
                sum1 *= -1

                sum2 = 0
                for j in range(i+1, self.task.n):
                    sum2 += result['approx'][j]*self.task.matrix(i, j)/self.task.matrix(i, i)
                sum2 *= -1

                sum3 = self.task.b(i)/self.task.matrix(i, i)

                next_approx.append(sum0 + param*(sum1 + sum2 + sum3))

            next_approx.append(0)

            result['approx'] = next_approx

        result['approx'] = result['approx'][1:-1]
        return result
