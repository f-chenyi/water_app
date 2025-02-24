from fenics import *
import numpy as np
import ufl
import copy
import os
import sys
import warnings
from matplotlib import pyplot as plt

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

warnings.filterwarnings("ignore")
parameters['reorder_dofs_serial'] = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["linear_algebra_backend"] = "PETSc"

solver = PETScTAOSolver()
solver.parameters["method"] = "tron"
solver.parameters["monitor_convergence"] = False
solver.parameters["report"] = False
solver.parameters["maximum_iterations"] = 30


app = Flask(__name__)
CORS(app)  # Enable CORS for development

LmeshX = 20
LmeshY = 20
NmeshX = 100
NmeshY = 100


class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        return bool(  ( near(x[0], -LmeshX) or near(x[1], -LmeshY) ) and
                      ( not ( (near(x[0], -LmeshX) and near(x[1],  LmeshY)) or
                      (near(x[0],  LmeshX) and near(x[1], -LmeshY)) )  )
                      and on_boundary )

    def map(self, x, y):
        if near(x[0], LmeshX) and near(x[1], LmeshY):
            y[0] = x[0] - 2*LmeshX
            y[1] = x[1] - 2*LmeshY
        elif near(x[0], LmeshX):
            y[0] = x[0] - 2*LmeshX
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 2*LmeshY


class WaterSimulation:

    def __init__(self, mu=0.01, tRev=1.0):
        # control params
        self.fnAvg = 0.1
        self.fnStd = 0.2*self.fnAvg
        self.tRev = tRev
        self.mu = mu
        self.Gamma = 1.0
        
        # cell size
        self.Rc = 1.0
    
        # steric interaction
        self.Bf = 8.0
        self.d_Bf = 0.2 * self.Rc
        
        # friction
        self.eta_n = 2e-4
        self.ETA = 0.85
        self.eta_p = self.eta_n * (1+self.ETA) / (1-self.ETA)
        
        # simulation params
        self.EPS_THETA = 1E-5
        self.EPS_DZ  = 2E-2
        self.RAND_DZ = 1E-2
        self.MAX_FAIL = 5
        self.MAX_DX  = self.d_Bf
        self.MIN_DT = 0.1
        self.Drot   = 1e-8
        self.dt = 0.005
        self.t  = 0.0
    
        # aux variables
        [SHIFT_X, SHIFT_Y] = np.meshgrid(np.linspace(-1,1,3).astype(int),np.linspace(-1,1,3).astype(int))
        self.SHIFT_X = np.reshape(SHIFT_X,(1,-1))[0]
        self.SHIFT_Y = np.reshape(SHIFT_Y,(1,-1))[0]


        # FEM
        self.mesh = RectangleMesh(Point(-LmeshX, -LmeshY), Point(LmeshX, LmeshY), NmeshX, NmeshY)
        self.pbc = PeriodicBoundary()
        self.dx   = Measure("dx", domain=self.mesh)
        
        self.V     = FunctionSpace(self.mesh, 'P', 1, constrained_domain=self.pbc)
        self.u     = Function(self.V)
        self.u0    = Function(self.V)
        self.u_min = Function(self.V)
        self.u_max = Function(self.V)
        self.u_max.vector()[:] = 2*self.Rc
        
        # Finite difference
        self.XX, self.YY = np.meshgrid( np.linspace(-LmeshX,LmeshX,NmeshX+1),
                                        np.linspace(-LmeshY,LmeshY,NmeshY+1))
        self.DxGrid = LmeshX * 2 / NmeshX
        self.DyGrid = LmeshY * 2 / NmeshY
        
        # cell variables
        self.pos_cell = np.loadtxt("init_10.txt")
        self.Num_cell = self.pos_cell.shape[0]
        LcArray = self.pos_cell[:, [3]]
        self.eta_cell = np.concatenate( (self.eta_n*2*LcArray, self.eta_p*2*LcArray, self.eta_p*2*LcArray**3/3), axis=1)
        self.magf_cell = self.fnAvg + self.fnStd*(2*np.random.rand(self.Num_cell) - 1.0)
        self.sgnf_cell = np.sign( np.random.rand(self.Num_cell)-0.5 )
        self.swtf_cell = np.random.exponential( self.tRev, self.Num_cell )
        self.pos_cell_out = [{"x": self.pos_cell[i,0], "y": self.pos_cell[i,1], "th": self.pos_cell[i,2], "l": self.pos_cell[i,3], "r": self.Rc, "sgn": self.sgnf_cell[i]} for i in range(self.Num_cell)]
        
        self.vel_all = np.zeros([self.Num_cell, 3])
        
        # water profile list
        self.water25 = []
        self.water50 = []
        self.water100 = []
        self.water150 = []
        
        # define FEM problem and solve the initial water profile
        self.assembleMinErgProb()
        self.solveMinErgProb()
        
    
    
    # ============================== FEM ===============================
    def assembleMinErgProb(self):
        
        u = self.u
        du = TrialFunction(self.V)
        v  = TestFunction(self.V)
        
        TotErg  = self.Gamma * (sqrt(1+inner(grad(self.u),grad(self.u))) - 1)*self.dx + self.mu * self.u * self.dx
        gradErg = derivative(TotErg, self.u, v)
        HesErg  = derivative(gradErg, self.u, du)

        class MinErgProblem(OptimisationProblem):

            def __init__(self):
                OptimisationProblem.__init__(self)

            # Objective function
            def f(self, x):
                u.vector()[:] = x
                return assemble(TotErg)
            # Gradient of the objective function
            def F(self, b, x):
                u.vector()[:] = x
                assemble(gradErg, tensor=b)
            # Hessian of the objective function
            def J(self, A, x):
                u.vector()[:] = x
                assemble(HesErg, tensor=A)
            
        self.problem = MinErgProblem()
        
        
    
    def solveMinErgProb(self):
        solver_count = 0
        self.updateUminVec();
        while solver_count <= self.MAX_FAIL:
            try:
                self.u.vector()[:] = self.u0.vector()
                solver.solve(self.problem, self.u.vector(), self.u_min.vector(), self.u_max.vector())
                break
            except RuntimeError:
                solver_count += 1
                self.u0.vector()[:] = self.u.vector().get_local() + self.RAND_DZ * ( np.random.random(np.shape(self.u0.vector().get_local())) )

    
    
    def getCellSurface(self, pos_o):
    
        z_o      = -self.Rc * np.ones(np.shape(self.XX))

        xc       = pos_o[0]
        yc       = pos_o[1]
        theta    = pos_o[2]
        clen     = pos_o[3]

        # define relevant distances
        S_PERP        = (self.XX - xc)*np.sin(theta) - (self.YY - yc)*np.cos(theta)
        S_PARA        = (self.XX - xc)*np.cos(theta) + (self.YY - yc)*np.sin(theta)
        CapDistSq_pos = (self.XX - xc - clen*np.cos(theta))**2 + (self.YY - yc - clen*np.sin(theta))**2
        CapDistSq_neg = (self.XX - xc + clen*np.cos(theta))**2 + (self.YY - yc + clen*np.sin(theta))**2

        # define relavant regions
        REG_SIDE  = np.abs(S_PERP) < self.Rc
        BODY_SIDE = REG_SIDE * (np.abs(S_PARA) < clen)
        CAP_PLUS  = (CapDistSq_pos < self.Rc**2)* (np.abs(S_PARA) >= clen)
        CAP_MINUS = (CapDistSq_neg < self.Rc**2)* (np.abs(S_PARA) >= clen)

        # top surface
        z_o[CAP_MINUS] = self.Rc + np.sqrt(self.Rc**2 - CapDistSq_neg[CAP_MINUS])
        z_o[CAP_PLUS]  = self.Rc  + np.sqrt(self.Rc**2 - CapDistSq_pos[CAP_PLUS])
        z_o[BODY_SIDE] = self.Rc + np.sqrt(self.Rc**2 - S_PERP[BODY_SIDE]**2)

        return z_o


    def near_x(self, pos_cell_o, x_test, Li, Ri):
        xc    = pos_cell_o[0]
        theta = pos_cell_o[2]
        return (np.abs(xc + Li*np.cos(theta) - x_test) < Ri) or (np.abs(xc - Li*np.cos(theta) - x_test) < Ri)\
                or ( np.sign(xc + Li*np.cos(theta) - x_test) != np.sign(xc - Li*np.cos(theta) - x_test) )

    def near_y(self, pos_cell_o, y_test, Li, Ri):
        yc    = pos_cell_o[1]
        theta = pos_cell_o[2]
        return (np.abs(yc + Li*np.sin(theta) - y_test) < Ri) or (np.abs(yc - Li*np.sin(theta) - y_test) < Ri)\
                or ( np.sign(yc + Li*np.sin(theta) - y_test) != np.sign(yc - Li*np.sin(theta) - y_test) )

    
    def outside_x(self, pos_cell_o):
        xc    = pos_cell_o[0]
        theta = pos_cell_o[2]
        clen   = pos_cell_o[3]
        if   ( xc + clen*np.cos(theta) >=  LmeshX+self.Rc ) and ( xc - clen*np.cos(theta) >=  LmeshX+self.Rc ):
            xx_shift = -1
        elif ( xc + clen*np.cos(theta) <= -LmeshX-self.Rc ) and ( xc - clen*np.cos(theta) <= -LmeshX-self.Rc ):
            xx_shift = +1
        else:
            xx_shift = 0
        return xx_shift
    
    
    def outside_y(self, pos_cell_o):
        yc    = pos_cell_o[1]
        theta = pos_cell_o[2]
        clen   = pos_cell_o[3]
        if   ( yc + clen*np.sin(theta) >=  LmeshY+self.Rc ) and ( yc - clen*np.sin(theta) >=  LmeshY+self.Rc ):
            yy_shift = -1
        elif ( yc + clen*np.sin(theta) <= -LmeshY-self.Rc ) and ( yc - clen*np.sin(theta) <= -LmeshY-self.Rc ):
            yy_shift = +1
        else:
            yy_shift = 0
        return yy_shift


    
    def shift_near_box(self, pos_cell_o, Li, Ri):

        x_shift = [0]
        y_shift = [0]
        if self.near_x(pos_cell_o, LmeshX, Li, Ri):
            x_shift.append(-1)
        if self.near_x(pos_cell_o,-LmeshX, Li, Ri):
            x_shift.append(+1)
        if self.near_y(pos_cell_o, LmeshY, Li, Ri):
            y_shift.append(-1)
        if self.near_y(pos_cell_o,-LmeshY, Li, Ri):
            y_shift.append(+1)

        return x_shift, y_shift


    def updateUminVec(self):

        region_cell_o = -self.Rc * np.ones(np.shape(self.XX))

        for i in range( self.Num_cell ):

            theta_tmp = self.pos_cell[i,2]
            Lc_tmp    = self.pos_cell[i,3]

            x_shift, y_shift = self.shift_near_box(self.pos_cell[i,[0,1,2]], Lc_tmp, 1.1*self.Rc)
            for xx in x_shift:
                for yy in y_shift:
                    xc_tmp = self.pos_cell[i,0] + xx * 2 * LmeshX
                    yc_tmp = self.pos_cell[i,1] + yy * 2 * LmeshY
                    region_cell_o = np.maximum( region_cell_o,\
                    self.getCellSurface(np.array([xc_tmp,yc_tmp, theta_tmp, Lc_tmp])) )

        region_cell_o = np.maximum(region_cell_o, 0.0)
        
        self.u_min.vector()[vertex_to_dof_map(self.V)] = np.reshape(region_cell_o, ((NmeshX+1)*(NmeshY+1),))

    # =================================================================
    
    
    # ============================== Cell forces ===============================
    def _closerArray(self, val1Array, val2Array, TARGET_VAL):
        valcArray = np.zeros(np.shape(val1Array))
        valcArray[:] = val1Array[:]
        ID_2  = np.abs(val2Array-TARGET_VAL) < np.abs(val1Array-TARGET_VAL)
        valcArray[ID_2] = val2Array[ID_2]
        return valcArray


    def _computedistArray(self, x1,y1,theta1,s1,x2,y2,theta2,s2):
        # all the input are interpreted as Array, and should be
        # conditioned to the correct shapes
        xm1 = x1 + s1*np.cos(theta1)
        ym1 = y1 + s1*np.sin(theta1)
        xm2 = x2 + s2*np.cos(theta2)
        ym2 = y2 + s2*np.sin(theta2)
        d   = np.sqrt( (xm1-xm2)*(xm1-xm2) + (ym1-ym2)*(ym1-ym2) )
        return d
    
    
    def _computeCellCellDistNOTPARA(self, POS_ALL_O, ix, iy):
    
        N_CELL   = np.shape(POS_ALL_O)[0]

        X1       = np.tile(np.reshape(POS_ALL_O[:,0],(N_CELL,1)), (1,N_CELL))
        X1       = X1 + ix*2*LmeshX
        Y1       = np.tile(np.reshape(POS_ALL_O[:,1],(N_CELL,1)), (1,N_CELL))
        Y1       = Y1 + iy*2*LmeshY
        THETA1   = np.tile(np.reshape(POS_ALL_O[:,2],(N_CELL,1)), (1,N_CELL))
        LC1      = np.tile(np.reshape(POS_ALL_O[:,3],(N_CELL,1)), (1,N_CELL))

        X2       = np.tile(                       POS_ALL_O[:,0], (N_CELL,1))
        Y2       = np.tile(                       POS_ALL_O[:,1], (N_CELL,1))
        THETA2   = np.tile(                       POS_ALL_O[:,2], (N_CELL,1))
        LC2      = np.tile(POS_ALL_O[:,3], (N_CELL,1))

        S1_INTERSECT = (  (X2-X1)*np.sin(THETA2) - (Y2-Y1)*np.cos(THETA2) ) / np.sin(THETA2-THETA1)
        S2_INTERSECT = (  (X1-X2)*np.sin(THETA1) - (Y1-Y2)*np.cos(THETA1) ) / np.sin(THETA1-THETA2)

        # end distance that are closer to the intersection point
        S1_END   = self._closerArray(LC1,-LC1,S1_INTERSECT)
        S2_END   = self._closerArray(LC2,-LC2,S2_INTERSECT)


        # smallest distance intersection
        S1_M    = np.zeros(np.shape(S1_END))
        S1_M[:] = S1_END[:]
        S2_M    = S1_END * np.cos(THETA2 - THETA1) + (X1-X2)*np.cos(THETA2) + (Y1-Y2)*np.sin(THETA2)
        S2_M_TMP= self._closerArray(LC2,-LC2,S2_M)
        S2_M[np.abs(S2_M) > LC2] = S2_M_TMP[np.abs(S2_M) > LC2]
        D_M     = self._computedistArray(X1,Y1,THETA1,S1_M, X2,Y2,THETA2,S2_M)


        # 2b. compute the distance if s2_m = s2_end
        S2_N    = np.zeros(np.shape(S2_END))
        S2_N[:] = S2_END[:]
        S1_N    = S2_END * np.cos(THETA2 - THETA1) + (X2-X1)*np.cos(THETA1) + (Y2-Y1)*np.sin(THETA1)
        S1_N_TMP= self._closerArray(LC1,-LC1,S1_N)
        S1_N[np.abs(S1_N) > LC1] = S1_N_TMP[np.abs(S1_N) > LC1]
        D_N     = self._computedistArray(X1,Y1,THETA1,S1_N, X2,Y2,THETA2,S2_N)

        S1_M[D_N < D_M] = S1_N[D_N < D_M]
        S2_M[D_N < D_M] = S2_N[D_N < D_M]
        D_M[D_N < D_M]  = D_N[D_N < D_M]

        return (D_M, S1_M, S2_M)



    # [+v6] compute minimal distances if cell I and cell J are parallel
    def _computeCellCellDistPARALLEL(self, POS_ALL_O, ix, iy):

        N_CELL   = np.shape(POS_ALL_O)[0]

        X1       = np.tile(np.reshape(POS_ALL_O[:,0],(N_CELL,1)), (1,N_CELL))
        X1       = X1 + ix*2*LmeshX
        Y1       = np.tile(np.reshape(POS_ALL_O[:,1],(N_CELL,1)), (1,N_CELL))
        Y1       = Y1 + iy*2*LmeshY
        THETA1   = np.tile(np.reshape(POS_ALL_O[:,2],(N_CELL,1)), (1,N_CELL))
        LC1      = np.tile(np.reshape(POS_ALL_O[:,3],(N_CELL,1)), (1,N_CELL))

        X2       = np.tile(                       POS_ALL_O[:,0], (N_CELL,1))
        Y2       = np.tile(                       POS_ALL_O[:,1], (N_CELL,1))
        THETA2   = np.tile(                       POS_ALL_O[:,2], (N_CELL,1))
        LC2      = np.tile(POS_ALL_O[:,3], (N_CELL,1))


        S1_2P = (X2 + LC2*np.cos(THETA2) - X1 )*np.cos(THETA1) + (Y2 + LC2*np.sin(THETA2) - Y1)*np.sin(THETA1)
        S1_2M = (X2 - LC2*np.cos(THETA2) - X1 )*np.cos(THETA1) + (Y2 - LC2*np.sin(THETA2) - Y1)*np.sin(THETA1)

        S2_1P = (X1 + LC1*np.cos(THETA1) - X2 )*np.cos(THETA2) + (Y1 + LC1*np.sin(THETA1) - Y2)*np.sin(THETA2)
        S2_1M = (X1 - LC1*np.cos(THETA1) - X2 )*np.cos(THETA2) + (Y1 - LC1*np.sin(THETA1) - Y2)*np.sin(THETA2)

        S1_M  = np.zeros([N_CELL,N_CELL])
        S2_M  = np.zeros([N_CELL,N_CELL])

        CASE_A = S1_2P>S1_2M
        S1_M[CASE_A] = 0.5*( np.minimum(S1_2P[CASE_A],LC1[CASE_A])+\
                             np.maximum(S1_2M[CASE_A],-LC1[CASE_A]) )
        S2_M[CASE_A] = 0.5*( np.minimum(S2_1P[CASE_A],LC2[CASE_A])+\
                             np.maximum(S2_1M[CASE_A],-LC2[CASE_A]) )
        CASE_Al = np.logical_and(S1_2P>S1_2M, S1_2P <= -LC1)
        S1_M[CASE_Al]= -LC1[CASE_Al]
        S2_M[CASE_Al]=  LC2[CASE_Al]
        CASE_Ar = np.logical_and(S1_2P>S1_2M, S1_2M >= LC1)
        S1_M[CASE_Ar]=  LC1[CASE_Ar]
        S2_M[CASE_Ar]= -LC2[CASE_Ar]

        CASE_B = S1_2P < S1_2M
        S1_M[CASE_B] = 0.5*( np.minimum(S1_2M[CASE_B], LC1[CASE_B])+\
                             np.maximum(S1_2P[CASE_B],-LC1[CASE_B]) )
        S2_M[CASE_B] = 0.5*( np.minimum(S2_1M[CASE_B], LC2[CASE_B])+\
                             np.maximum(S2_1P[CASE_B],-LC2[CASE_B]) )
        CASE_Bl= np.logical_and(S1_2P < S1_2M, S1_2M <= -LC1)
        S1_M[CASE_Bl]= -LC1[CASE_Bl]
        S2_M[CASE_Bl]= -LC2[CASE_Bl]
        CASE_Br= np.logical_and(S1_2P < S1_2M, S1_2P >= LC1)
        S1_M[CASE_Br]= LC1[CASE_Br]
        S2_M[CASE_Br]= LC2[CASE_Br]

        D_M     = self._computedistArray(X1,Y1,THETA1,S1_M, X2,Y2,THETA2,S2_M)

        return (D_M, S1_M, S2_M)
    
    
    def computeCellCellForce(self, POS_ALL_O):

        N_CELL        = np.shape(POS_ALL_O)[0]
        THETA1        = np.tile(np.reshape(POS_ALL_O[:,2],(N_CELL,1)), (1,N_CELL))
        THETA2        = np.tile(                       POS_ALL_O[:,2], (N_CELL,1))

        D_M           = 10*LmeshX*np.ones([N_CELL,N_CELL])
        S1_M          = np.zeros([N_CELL,N_CELL])
        S2_M          = np.zeros([N_CELL,N_CELL])
        OPT_SHIFT_X1  = np.zeros([N_CELL,N_CELL])
        OPT_SHIFT_Y1  = np.zeros([N_CELL,N_CELL])

        for i in range(len(self.SHIFT_X)):

            D_M_this,      S1_M_this,      S2_M_this      = self._computeCellCellDistNOTPARA(POS_ALL_O, self.SHIFT_X[i], self.SHIFT_Y[i])
            D_M_PARA_this, S1_M_PARA_this, S2_M_PARA_this = self._computeCellCellDistPARALLEL(POS_ALL_O, self.SHIFT_X[i], self.SHIFT_Y[i])

            D_M_this[np.abs(np.sin(THETA1-THETA2)) < self.EPS_THETA]  = D_M_PARA_this[np.abs(np.sin(THETA1-THETA2)) < self.EPS_THETA]
            S1_M_this[np.abs(np.sin(THETA1-THETA2)) < self.EPS_THETA] = S1_M_PARA_this[np.abs(np.sin(THETA1-THETA2)) < self.EPS_THETA]
            S2_M_this[np.abs(np.sin(THETA1-THETA2)) < self.EPS_THETA] = S2_M_PARA_this[np.abs(np.sin(THETA1-THETA2)) < self.EPS_THETA]

            S1_M[D_M_this < D_M] = S1_M_this[D_M_this < D_M]
            S2_M[D_M_this < D_M] = S2_M_this[D_M_this < D_M]
            OPT_SHIFT_X1[D_M_this < D_M] = self.SHIFT_X[i]
            OPT_SHIFT_Y1[D_M_this < D_M] = self.SHIFT_Y[i]
            D_M[D_M_this < D_M]  = D_M_this[D_M_this < D_M]


        X1            = np.tile(np.reshape(POS_ALL_O[:,0],(N_CELL,1)), (1,N_CELL))
        X1            = X1 + OPT_SHIFT_X1*2*LmeshX
        Y1            = np.tile(np.reshape(POS_ALL_O[:,1],(N_CELL,1)), (1,N_CELL))
        Y1            = Y1 + OPT_SHIFT_Y1*2*LmeshY

        X2            = np.tile(                       POS_ALL_O[:,0], (N_CELL,1))
        Y2            = np.tile(                       POS_ALL_O[:,1], (N_CELL,1))

        # in-plane unit vector connecting the closest-distance pair
        # (from 2 to 1)
        n1xArray = ( (X1+S1_M*np.cos(THETA1)) - (X2+S2_M*np.cos(THETA2)) ) / D_M
        n1yArray = ( (Y1+S1_M*np.sin(THETA1)) - (Y2+S2_M*np.sin(THETA2)) ) / D_M

        # cell-cell force magnitude
        fmagArray = self.Bf * (0.5 - 0.5*np.tanh( (D_M - 2*self.Rc) /self.d_Bf))
        np.fill_diagonal(fmagArray, 0.0)

        FX_ALL      = fmagArray*n1xArray
        FY_ALL      = fmagArray*n1yArray
        TAU_ALL     = S1_M*np.cos(THETA1)*FY_ALL - S1_M*np.sin(THETA1)*FX_ALL

        np.fill_diagonal(FX_ALL, 0.0)
        np.fill_diagonal(FY_ALL, 0.0)
        np.fill_diagonal(TAU_ALL, 0.0)

        return FX_ALL, FY_ALL, TAU_ALL
    
    # ==========================================================================
    
    # ========================= compute water force ============================
    
    def _gradRoll(self, phi, dxi, dyi):
        
        phi_n = phi[0:-1,0:-1]

        phir = np.roll(phi_n, -1, axis=1)
        phil = np.roll(phi_n,  1, axis=1)
        phid = np.roll(phi_n, -1, axis=0)
        phiu = np.roll(phi_n,  1, axis=0)

        phi_x = 0.5*(phir-phil)/dxi
        phi_y = 0.5*(phid-phiu)/dyi

        # recover the same size of the array
        phi_x = np.concatenate( (phi_x,np.array([phi_x[:,0]]).T), axis = 1 )
        phi_x = np.concatenate( (phi_x,np.array([phi_x[0,:]])  ), axis = 0 )
        phi_y = np.concatenate( (phi_y,np.array([phi_y[:,0]]).T), axis = 1 )
        phi_y = np.concatenate( (phi_y,np.array([phi_y[0,:]])  ), axis = 0 )

        return(phi_x, phi_y)

    def _divRoll(self, ux, uy, dxi, dyi):
    
        ux_n = ux[:,0:-1]
        uy_n = uy[0:-1,:]

        uxr = np.roll(ux_n, -1, axis=1)
        uxl = np.roll(ux_n,  1, axis=1)

        uyd = np.roll(uy_n, -1, axis=0)
        uyu = np.roll(uy_n,  1, axis=0)

        ux_x = 0.5 * (uxr - uxl) / dxi
        uy_y = 0.5 * (uyd - uyu) / dyi

        ux_x = np.concatenate( (ux_x,np.array([ux_x[:,0]]).T), axis=1 )
        uy_y = np.concatenate( (uy_y,np.array([uy_y[0,:]])  ), axis=0 )
    
        return ux_x + uy_y


    def computeWaterForce(self, fx_o, fy_o, tau_o, pos_o):
    
        z_water = np.reshape(
                    self.u.vector()[vertex_to_dof_map(self.V)], (NmeshY+1,NmeshX+1)
                    )

        z_x, z_y = self._gradRoll(z_water, self.DxGrid, self.DyGrid)
        zfunc_x    = z_x / np.sqrt(1 + z_x*z_x + z_y*z_y)
        zfunc_y    = z_y / np.sqrt(1 + z_x*z_x + z_y*z_y)

        # force density (see Eqns in our manuscript)
        fx_mat  = - self.Gamma * z_x * self._divRoll(zfunc_x, zfunc_y, self.DxGrid, self.DyGrid) + self.mu * z_x
        fy_mat  = - self.Gamma * z_y * self._divRoll(zfunc_x, zfunc_y, self.DxGrid, self.DyGrid) + self.mu * z_y

        # compute forces on all cells
        for cell_i in range(pos_o.shape[0]):

            theta_tmp = pos_o[cell_i,2]
            Lc_tmp    = pos_o[cell_i,3]

            # reset force array
            fx_o[cell_i][cell_i]  = 0.0
            fy_o[cell_i][cell_i]  = 0.0
            tau_o[cell_i][cell_i] = 0.0

            x_shift, y_shift = self.shift_near_box(pos_o[cell_i,[0,1,2]],Lc_tmp, self.Rc)
            for xx in x_shift:
                for yy in y_shift:
                    xc_tmp   = pos_o[cell_i][0] + xx * 2 * LmeshX
                    yc_tmp   = pos_o[cell_i][1] + yy * 2 * LmeshY

                    # define contact region
                    z_surf   = self.getCellSurface(np.array([xc_tmp, yc_tmp, theta_tmp, Lc_tmp]))
                    cont_mat = ( z_surf >= 0.5*self.Rc ) * (z_water - z_surf < self.EPS_DZ)
                    cont_mat[:,NmeshX]=False
                    cont_mat[NmeshY,:]=False

                    # define cell-body axis coordinate
                    xcont_mat = self.XX[cont_mat]
                    ycont_mat = self.YY[cont_mat]

                    # compute tangent force
                    fx_tan    = fx_mat[cont_mat]
                    fy_tan    = fy_mat[cont_mat]
                    tau_tan   = (xcont_mat-xc_tmp)*fy_tan\
                              - (ycont_mat-yc_tmp)*fx_tan

                    # integral on the grid
                    fx_o[cell_i][cell_i]  += sum( fx_tan )  * (self.DxGrid * self.DyGrid)
                    fy_o[cell_i][cell_i]  += sum( fy_tan )  * (self.DxGrid * self.DyGrid)
                    tau_o[cell_i][cell_i] += sum( tau_tan ) * (self.DxGrid * self.DyGrid)
                    
    # ==========================================================================
    
    def updateCellVel(self, fx_all, fy_all, tau_all):
        
        theta_all = self.pos_cell[:,2]

        for cell_i in range(self.Num_cell):

            nx_i  = np.cos(theta_all[cell_i])
            ny_i  = np.sin(theta_all[cell_i])
            fpara = np.sum(fx_all[cell_i][:])*nx_i + np.sum(fy_all[cell_i][:])*ny_i
            fperp =-np.sum(fx_all[cell_i][:])*ny_i + np.sum(fy_all[cell_i][:])*nx_i

            self.vel_all[cell_i][0] = fpara*nx_i/self.eta_cell[cell_i,0] - fperp*ny_i/self.eta_cell[cell_i,1] +\
                                 self.sgnf_cell[cell_i]*self.magf_cell[cell_i]*nx_i/self.eta_cell[cell_i,0]
            self.vel_all[cell_i][1] = fperp*nx_i/self.eta_cell[cell_i,1] + fpara*ny_i/ self.eta_cell[cell_i,0] +\
                                 self.sgnf_cell[cell_i]*self.magf_cell[cell_i]*ny_i/self.eta_cell[cell_i,0]
            self.vel_all[cell_i][2] = np.sum(tau_all[cell_i][:])/self.eta_cell[cell_i,2]
            
    # determine the adaptive time increment
    def getDt(self):

        v_trans = np.sqrt( self.vel_all[:,0]**2 + self.vel_all[:,1]**2 )
        v_rot   = self.vel_all[:,2]
        LcArray = self.pos_cell[:,3]
        
        Dt_trans= self.MAX_DX/np.amax(v_trans)
        Dt_rot  = self.MAX_DX/np.amax(np.abs(v_rot)*LcArray)
        Dt      = np.amin( [Dt_trans, Dt_rot, self.MIN_DT] )

        return Dt

    
    def scale2out(self):
        # canvas = 500 * 500
        xyfactor = 500 / LmeshX / 2
        for i in range(self.Num_cell):
            self.pos_cell_out[i]["x"] = ( self.pos_cell[i, 0] + LmeshX ) * xyfactor
            self.pos_cell_out[i]["y"] = (self.pos_cell[i, 1] + LmeshY ) * xyfactor
            self.pos_cell_out[i]["th"] = self.pos_cell[i, 2]
            self.pos_cell_out[i]["l"] = self.pos_cell[i, 3] * xyfactor
            self.pos_cell_out[i]["r"] = self.Rc * xyfactor
            self.pos_cell_out[i]["sgn"] = self.sgnf_cell[i]
            
    # only works for matplotlib < 3.8
#    def water2out(self):
#        xyfactor = 500 / LmeshX / 2
#        contours =  plt.contour(self.XX, self.YY,
#            np.reshape(self.u.vector()[vertex_to_dof_map(self.V)], (NmeshY+1,NmeshX+1)),
#            levels=[ 0.25, 0.5, 1.0, 1.5]
#        )
#        self.water25 = [{"x": (p.vertices[:,0] + LmeshX) * xyfactor, "y": (p.vertices[:,1] + LmeshY) * xyfactor } for p in contours.collections[0].get_paths()]
#        self.water50 = [{"x": (p.vertices[:,0] + LmeshX) * xyfactor, "y": (p.vertices[:,1] + LmeshY) * xyfactor } for p in contours.collections[1].get_paths()]
#        self.water100 = [{"x": (p.vertices[:,0] + LmeshX) * xyfactor, "y": (p.vertices[:,1] + LmeshY) * xyfactor } for p in contours.collections[2].get_paths()]
#        self.water150 = [{"x": (p.vertices[:,0] + LmeshX) * xyfactor, "y": (p.vertices[:,1] + LmeshY) * xyfactor } for p in contours.collections[3].get_paths()]

    def water2out(self, levels=[0.25, 0.5, 1.0, 1.5]):
        xyfactor = 500 / LmeshX / 2
        contours =  plt.contour(self.XX, self.YY,
            np.reshape(self.u.vector()[vertex_to_dof_map(self.V)], (NmeshY+1,NmeshX+1)),
            levels=levels
        )

        ind = []
        for i in range(len(levels)):
            if len(contours.get_paths()[i]) == 0:
                ind.append([])
            else:
                startp = np.append( np.where( contours.get_paths()[i].codes == 1)[0], len(contours.get_paths()[i].codes) )
                ind.append(startp)
        
        self.water25 = [{"x": list( (contours.get_paths()[0].vertices[ind[0][i]:ind[0][i+1],0] + LmeshX) * xyfactor ) ,
                         "y": list( (contours.get_paths()[0].vertices[ind[0][i]:ind[0][i+1],1] + LmeshY) * xyfactor ) }
                        for i in range(len(ind[0])-1)]
        self.water50 = [{"x": list( (contours.get_paths()[1].vertices[ind[1][i]:ind[1][i+1],0] + LmeshX) * xyfactor ) ,
                         "y": list( (contours.get_paths()[1].vertices[ind[1][i]:ind[1][i+1],1] + LmeshY) * xyfactor ) }
                        for i in range(len(ind[1])-1)]
        self.water100 = [{"x": list( (contours.get_paths()[2].vertices[ind[2][i]:ind[2][i+1],0] + LmeshX) * xyfactor ) ,
                          "y": list( (contours.get_paths()[2].vertices[ind[2][i]:ind[2][i+1],1] + LmeshY) * xyfactor ) }
                        for i in range(len(ind[2])-1)]
        self.water150 = [{"x": list( (contours.get_paths()[3].vertices[ind[3][i]:ind[3][i+1],0] + LmeshX) * xyfactor ) ,
                          "y": list( (contours.get_paths()[3].vertices[ind[3][i]:ind[3][i+1],1] + LmeshY) * xyfactor ) }
                        for i in range(len(ind[3])-1)]
        
    
    def step(self):
        
        dt_total = 0.0
        breakflag = False
        
        while True:
            # record old water profile
            self.u0.vector()[:] = self.u.vector()

            # clear velocity matrix
            self.vel_all[:] = 0.

            # compute water force
            fx_all, fy_all, tau_all = self.computeCellCellForce(self.pos_cell)

            # compute cell force
            self.computeWaterForce(fx_all, fy_all, tau_all, self.pos_cell)

            # compute velocity
            self.updateCellVel(fx_all, fy_all, tau_all)
            
            dt_this = self.getDt();
            
            if dt_total + dt_this > self.dt:
                breakflag = True
                dt_this = self.dt - dt_total
            
            dt_total += dt_this
            self.t += dt_this
            
            # update active cell driving
            if np.any(self.swtf_cell<self.t):
                self.sgnf_cell[ self.swtf_cell<self.t] = - self.sgnf_cell[self.swtf_cell<self.t]
                self.swtf_cell[ self.swtf_cell<self.t ] = self.t + np.random.exponential( self.tRev, np.sum(self.swtf_cell<self.t) )

            # update cell positions
            self.pos_cell[:,0:-1] += dt_this*self.vel_all
            self.pos_cell[:, 2] += np.sqrt(dt_this) * np.random.normal( 0.0, np.sqrt( self.Drot ), self.pos_cell[:,2].shape )
            
            # if cells are out of domain move back
            for cell_i in range(self.Num_cell):
                xx_shift = self.outside_x( self.pos_cell[cell_i][:] )
                yy_shift = self.outside_y( self.pos_cell[cell_i][:] )
                self.pos_cell[cell_i,[0,1]] += np.array([xx_shift * 2.0 * LmeshX, yy_shift * 2.0 * LmeshY])

            # update water
            self.assembleMinErgProb()
            self.solveMinErgProb()
            
            if breakflag:
                break


simulation = WaterSimulation(mu=0.02, tRev=0.5)

@app.route('/simulate_step')
def simulate_step():
    # retrieve params from query params
    mu = float(request.args.get('capillaryStrength', simulation.mu))
    tRev = float(request.args.get('meanReversalTime', simulation.tRev))
    
    # update simulation params
    simulation.mu = mu
    if simulation.tRev != tRev:
        simulation.swtf_cell = simulation.t + np.random.exponential( tRev, simulation.Num_cell )
    simulation.tRev = tRev
    
    simulation.step()
    simulation.scale2out()
    simulation.water2out()
    
    # output
    out = {
    "particles": simulation.pos_cell_out,
    "water25": simulation.water25,
    "water50": simulation.water50,
    "water100": simulation.water100,
    "water150": simulation.water150
    }
    
    return jsonify(out)
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
    
    
    
    
