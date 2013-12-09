import QP
import numpy as np
import Constants
import util
#import ipdbc


class GQP(QP.QP):
    '''
    GQP is a QP that is prepared to be solved using one of
    Goldfarb's primal or dual methods.
    '''
    def __init__(self, L, a, CT, b, ne=0):
        QP.QP.__init__(self, L, a, CT, b)
        # list of integers, the working set
        self.A = []
        # J = L^{-T}Q
        self.J = self.L.T.I
        # vector, dual variables
        self.u = None
        ## L^{-1}N = [Q_1 | Q_2] [R | 0]^T
        self.R = None
        # number of equality variables, the coefficients are
        # in the first ne rows of CT
        self.ne = ne
        self.numberOfActiveSetChanges = 0

        self.x = self.getUnconstrainedSol()
        self.f = 0.5 * self.a.T * self.x
        if ne == 0:
            self.x = np.matrix(self.x, Constants.DATATYPE)

        for i in range(ne):
            n_new = self.CT[i, :].T
            z = self.J2 * self.J2.T * n_new

            s = -self.sSome(self.x, [i])
            t2 = 0
            if (z.T * n_new) >= Constants.EPSILON:
                t2 = (s / (z.T * n_new))[0, 0]

            self.x += t2 * z
            self.f += 0.5 * (t2 * t2) * (z.T * n_new)

            try:
                self.u = np.concatenate((self.u, np.matrix([[t2]])))
            except:
                self.u = np.matrix([[t2]])

            if self.q > 0:
                r = np.linalg.solve(self.R, self.getd1(i))
                self.u[:self.q] -= t2 * r
            self.addConstraint(i)

    def sSome(self, x, subsetOfCons):
        m = [self.CT[i, :] * x - self.b[i] for i in subsetOfCons]
        return np.concatenate(m)

    def getq(self):
        '''
        getter for property q, the number of active constraints
        '''
        return len(self.A)

    q = property(getq)

    def getMostViolatedConst(self, x):
        '''
        returns the index of the most violated constraint by the point x
        '''
        s = self.sAll(x)

        # Set the slack of the active constraints to
        # exact zero to make sure they are never selected
        for i in range(self.m):
            if i in self.A:
                s[i] = 0
        minInd = s.argmin()
        if s[minInd] > -Constants.EPSILON:
            return -1
        return s.argmin()

    def getJ1(self):
        '''
        Return J_1 the first q columns of #J as a n*q matrix
        '''
        return self.J[:, :self.q]

    J1 = property(getJ1)

    def getJ2(self):
        '''
        returns J_2, the last n-q columns of #J as a n*(n-q) matrix
        '''
        return self.J[:, self.q:]

    J2 = property(getJ2)

    def getd1(self, i):
        return self.J1.T * self.CT[i, :].T

    d1 = property(getd1)

    def getd2(self, i):
        return self.J2.T * self.CT[i, :].T

    d2 = property(getd2)

    def getDualDirect_Step(self, r):
        '''
        - append a 1 to the end of -r to get the dual direction (DAS)
        - If the working set is empty, returns a vector with a
        single element = 1
        '''
        if self.q > 0:
            return np.concatenate((-r, np.matrix([[1.]])))
        return np.matrix([[1]])

    def checkInvars(self):
        ret = True
        if self.q > 0:
            Q1, R = np.linalg.qr(self.L.I *
                                (self.CT[[i for i in self.A], :]).T)

            # Make sure that Q2 and J2 are alright
            if (abs((self.L.T * self.J2).T * Q1) > 0.1).any() or \
                (abs(Q1.T * (self.L.T * self.J2)) > 0.1).any():
                #print "post1" , abs((self.L.T * self.J2).T * Q1)
                #print abs(Q1.T * (self.L.T * self.J2))
                #print "--"
                ret = False

            # Make sure that R is OK and N_star IS the pseudo-inverse of N
            N_star_N = (R.I * self.J1.T) * (self.CT[[i for i in self.A], :].T)
            iden = np.matrix(np.identity(self.q, float))

            if (abs(N_star_N) - iden > 0.5).any():
                #print "post2" , abs(N_star_N) - iden
                #print "--"
                ret = False
        return ret

    def addConstraint(self, i, d1=None, d2=None):
        '''
        Adds constraint i to the working set A by:
            - Using the Givens transformations to update J and R
            - Appending index i to the end of A
        '''
        try:
            d1.shape
        except:
            d1 = self.getd1(i)
            d2 = self.getd2(i)
        Ql, Q_bar, delta = util.applyGivens(d2)
        self.J = np.concatenate((self.J1, self.J2 * Q_bar.T), axis=1)
        if self.q == 0:
            self.R = np.matrix([[delta]], Constants.DATATYPE)
        else:
            self.R = np.concatenate((self.R, d1), axis=1)
            lastline = np.matrix([[0 for k in range(self.q)] + [delta]])
            self.R = np.concatenate((self.R, lastline))

        self.A.append(i)
        self.numberOfActiveSetChanges += 1
        return Q_bar

    def removeConstraint(self, l, uPlus):
        '''Removes constraint number l from the working set #A:
             -Removes the corresponding dual variable from uPlus
             -Updates J and R
             -Removes the l'th element of A
        '''
        q = self.q
        if l == 0:
            uPlus = uPlus[1:, 0]
        else:
            u_l = uPlus[l, 0]
            uPlus = np.concatenate((uPlus[:l, 0], uPlus[(l + 1):, 0]))

        if q == 1:
            self.R = None
            #TODO: Verify
            #self.J = self.L.T.I
        elif l == q - 1:
            self.R = self.R[:(q - 1), :(q - 1)]
        else:
            T = self.R[l:, (l + 1):]
            Q_bar, R2 = util.UH2UT(T)
            if l == 0:
                self.R = R2
            else:
                #Add zeros to R2 to make the lower q-1-l rows of R_minus
                R2 = np.concatenate((np.matrix(
                                       np.zeros((q - 1 - l, l))), R2), axis=1)
                #Remove the l'th column of R
                self.R = np.concatenate((self.R[:, :l],
                                         self.R[:, (l + 1):]), axis=1)
                # Construct R_minus from the upper l rows of R
                #  and q-1-l rows of R2 constructed above
                self.R = np.concatenate((self.R[:l, :], R2))

            #constructing J_minus
            mat_temp = np.matrix(np.identity(self.n, Constants.DATATYPE))
            mat_temp[l:q, l:q] = Q_bar.T
            self.J *= mat_temp

        self.A.pop(l)
        self.numberOfActiveSetChanges += 1

        if q == 1 or l == q - 1:
            return uPlus, 1

        return uPlus, mat_temp[:q, :q]

    def minStep(self, t1, t2):
        '''
        Returns the minimum of non-negative t1 and t2
        considering -1 to denote infinity
        '''
        m = min(t1, t2)
        if m == -1:
            if t1 != -1:
                return t1
            return t2
        return m

    def getMinRatio(self, uPlus, r, sign=1):
        '''
        Return min{u_plus[j] / r[j]} for (r[j] * sign) > 0
        '''
        m = mInd = -1
        for j in range(self.ne, r.shape[0]):
            if r[j, 0] * sign > Constants.EPSILON:
                ratio = uPlus[j, 0] / r[j, 0]
                if ratio < m or m == -1:
                    m = ratio
                    mInd = j
        return m, mInd

    def dual(self, verbose):
        '''
        Goldfarb dual active-set algorithm for convex QP
        '''
        self.__init__(self.L, self.a, self.CT, self.b, self.ne)

        iteration = 0
        while True:
            iteration += 1
            if verbose:
                print '-----------------------------------------'
                print 'iteration = ', iteration

                #print "obj value=" , f
                print "active set:"
                print self.A
            p = self.getMostViolatedConst(self.x)

            if p == -1:
                return 'feasible', self.x, self.f[0, 0]

            if self.q == 0:
                self.u = np.matrix([[0.]], Constants.DATATYPE)
                uPlus = np.matrix([[0.]], Constants.DATATYPE)
            else:
                uPlus = np.concatenate((self.u, np.matrix([[0.]])))

            n_new = self.CT[p, :].T

            #step2: partial step
            while True:
                #step 2.a)
                z = self.J2 * self.J2.T * n_new
                dual_dir = None
                if self.q > 0:
                    'calculate r'
                    r = np.linalg.solve(self.R, self.getd1(p))
                    dual_dir = np.concatenate((-r, np.matrix([[1.]],
                                              Constants.DATATYPE)))
                else:
                    dual_dir = np.matrix([[1]], Constants.DATATYPE)

                #step 2.b)
                t1 = t2 = -1
                #Partial Step Length
                if self.q > 0 and (r > 0).any():
                    t1, k = self.getMinRatio(uPlus, r)

                #Full Step Length
                if (abs(z) > Constants.EPSILON).any():
                    t2 = (-self.sSome(self.x, [p]) / (z.T * n_new))[0, 0]

                t = self.minStep(t1, t2)

                #step 2.c)
                if t == -1:
                    return 'infeasible', -1, -1
                elif t2 == -1:
                    if verbose:
                        print "taking a partial step_1"
                    uPlus += t * dual_dir
                    #we'll not use Q_tilde here. It is used in the PAS method.
                    uPlus, Q_tilde = self.removeConstraint(k, uPlus)
                    continue
                self.x += t * z

                #Not sure if it's smart to update f at each iteration
                self.f += t * z.T * n_new * (0.5 * t + uPlus[-1, 0])
                uPlus += t * dual_dir

                if abs(t - t2) <= Constants.EPSILON:
                    if verbose:
                        print "taking a full step "
                        print 'blocking constriant : ', p
                    self.u = uPlus
                    self.addConstraint(p)
                    #We took the full step, going back to step 1
                    break
                else:
                    if verbose:
                        print "taking a partial step_2"
                    #t == t1
                    uPlus, Q_tilde = self.removeConstraint(k, uPlus)
                    continue

    def primal(self, x, A, verbose):
        '''
        Goldfarb Primal active-set algorithm for convex QP
        '''
        for activeConstraint in A:
            self.addConstraint(activeConstraint)

        u = np.linalg.solve(self.R, self.J1.T * self.gradient(x))

        iteration = 0
        while True:
            iteration += 1
            if self.q > 0:
                p = u.argmin()
                IndOfConstraintToBeRemoved = self.A[p]
                if u[p] >= -Constants.EPSILON:
                    return 1, x, 0

                u_p = u[p, 0]
                R_p = self.R[:, p]

                u, Q_tilde = self.removeConstraint(p, u)

                d = Q_tilde * (u_p * R_p)

                d1 = d[:-1, 0]
                d2 = np.zeros((self.n - self.q, 1))
                d2[0, 0] = d[-1, 0]

            while self.q < self.n:

                #2
                if self.q < self.n:
                    z = -self.J2 * d2
                else:
                    z = 0

                if self.q > 0:
                    r = np.linalg.solve(self.R, d1)

                t1 = 1

                nonActive = [k for k in range(self.m) if k not in self.A]
                v = np.concatenate([self.CT[i, :] * z for i in nonActive], 1).T
                s = self.sSome(x, nonActive)

                m, mInd = self.getMinRatio(-s, v, -1)

                t1 = 1
                if m != -1:
                    t1 = min(1, m)

                x += t1 * z
                if self.q > 0:
                    u += t1 * r

                if t1 != 1:
                    Q_bar = self.addConstraint(nonActive[mInd])
                    u = np.concatenate((u, np.matrix([[0.]])))
                    d22 = self.J2.T * self.CT[0, :].T
                    d = (1 - t1) * d
                    d1 = d[:(self.q), 0]
                    d2 = d[(self.q):, 0]
                else:
                    break

        return 0, 0, 0
