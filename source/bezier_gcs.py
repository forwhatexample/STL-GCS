from pydrake.all import *
import warnings
import numpy as np
from math import floor
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.optimize import root_scalar
from pathlib import Path
import time
import math
import builtins

from experiment_reporting import parse_mosek_log, solver_tolerances

class BezierGraphOfConvexSets():
    """
    Problem setup and solver for planning a piecewise bezier curve trajectory
    through a graph of convex sets. The graph setup is as follows:

        - Each vertex is associated with a convex set
        - Each convex set contains a Bezier curve
        - The optimal path is a sequence of Bezier curves. These curves
          must line up with each other. 
        - The goal is to find a (minimum cost) trajectory from a given starting
          point to the target vertex
        - The target vertex is not associated with any constraints on the
          curve: it just indicates that the task is complete.
    """
    def __init__(self, vertices, edges, regions, start_vertex, end_vertex, start_point,
                 reset_information, time_constraint_information, end_time, vertices_time_bound, ts_states,
                 vertices_activate_var, order=2, continuity=1, lower_bound = None, upper_bound = None,
                 min_segment_duration=0.25):
        """
        Construct a graph of convex sets

        Args:
            vertices: list of integers representing each vertex in the graph
            edges: list of pairs of integers (vertices) for each edge
            regions: dictionary mapping each vertex to a Drake ConvexSet
            start_vertex: index of the starting vertex
            end_vertex: index of the end/target vertex
            start_point: initial point of the path
            real_time_index: the index of
            reset_information: a map from edge to the reset time variables
            time_constraint_information: a map from edge to its time constraints
            end_time: time horizon of task
            order: order of bezier curve under consideration
            continuity: number of continuous derivatives of the curve
            min_segment_duration: lower bound on the physical duration of each
                segment when lower/upper velocity bounds are enforced
        """
        # Dimensionality of the problem is defined by the starting point
        #assert regions[start_vertex].PointInSet(start_point)
        self.vertices = vertices
        self.edges = edges
        self.start_point = start_point
        self.dim = len(start_point)
        first_edge = list(time_constraint_information.keys())[0]
        self.time_var_list = list(time_constraint_information[first_edge].keys())
        self.time_var_list.remove('c0')
        self.end_time = end_time
        self.ts_states = ts_states
        assert min_segment_duration >= 0.0
        self.min_segment_duration = min_segment_duration
        self.cost_specifications = []
        self.velocity_bounds = {
            'lower': None if lower_bound is None else list(lower_bound),
            'upper': None if upper_bound is None else list(upper_bound),
        }

        # Check that the regions correspond to valid vertices and valid convex
        # sets
        for vertex, region in regions.items():
            assert vertex in self.vertices, "invalid vertex index"
            assert isinstance(region, ConvexSet), "regions must be convex sets"
            assert region.ambient_dimension() == self.dim
        self.regions = regions

        # Validate the start and target vertices
        assert start_vertex in vertices
        assert end_vertex in vertices
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
       
        # Bezier curves can guarantee continuity of n-1 derivatives
        assert continuity < order
        self.order = order
        self.continuity = continuity

        # Create "dummy" symbolic curves for an arbitrary edge. This allows us
        # to derive expressions for various things, such as derivatives of the
        # spline, in terms of the original control points (decision variables)
        self.u_control = MakeMatrixContinuousVariable(self.order+1, self.dim, "xu")
        self.v_control = MakeMatrixContinuousVariable(self.order+1, self.dim, "xv")
        self.u_duration = MakeVectorContinuousVariable(order + 1, "Tu_d")
        self.v_duration = MakeVectorContinuousVariable(order + 1, "Tv_d")
        self.u_timevar = MakeVectorContinuousVariable(len(self.time_var_list), "Tu_v")
        self.v_timevar = MakeVectorContinuousVariable(len(self.time_var_list), "Tv_v")


        self.u_vars = np.concatenate((self.u_control.flatten(),  self.u_duration, self.u_timevar))
        # print(self.u_vars)

        self.edge_vars = np.concatenate((
            self.u_control.flatten(),  self.u_duration, self.u_timevar,
            self.v_control.flatten(),  self.v_duration, self.v_timevar))

        self.bezier_path_u = BsplineTrajectory_[Expression](
                BsplineBasis_[Expression](self.order+1, self.order+1, 
                    KnotVectorType.kClampedUniform, 0, 1),
                self.u_control.T)
        self.bezier_path_v = BsplineTrajectory_[Expression](
                BsplineBasis_[Expression](self.order+1, self.order+1, 
                    KnotVectorType.kClampedUniform, 0, 1),
                self.v_control.T)

        self.bezier_duration_u = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](self.order + 1, self.order + 1,
                                      KnotVectorType.kClampedUniform, 0, 1),
            np.expand_dims(self.u_duration, 0))
        self.bezier_duration_v = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](self.order + 1, self.order + 1,
                                      KnotVectorType.kClampedUniform, 0, 1),
            np.expand_dims(self.v_duration, 0))

        # Create the GCS problem
        self.gcs = GraphOfConvexSets()
        #time_var_to_int = dict()
        #time_var_list = []
        #num = 0
        # for time_var in time_constraint_information[first_edge].keys():
        #     time_var_to_int[time_var] = num
        #     time_var_list.append(time_var)
        #     num += 1
        self.source, self.target = self.SetupShortestPathProblem(reset_information, time_constraint_information, vertices_time_bound, vertices_activate_var, lower_bound, upper_bound)

    def AddLengthCost(self, weight=1.0, norm="L2"):
        """
        Add a penalty on the distance between control points, which is an
        overapproximation of total path length.

        There are several norms we can use to approximate these lenths:
            L1 - most computationally efficient, and introduces only linear
            costs and constraints to the GCS problem.

            L2 - closest approximation to the actual curve length, introduces
            cone constraints to the GCS problem.

            L2_squared - incentivises evenly space control points, which can
            produce some nice smooth-looking curves. Introduces cone constraints
            to the GCS problem. 

        Args:
            weight: Weight for this cost, scalar.
            norm: Norm to use to when evaluating distance between control
                  points. See AddDerivativeCost for details.
        """
        assert norm in ["L1", "L2", "L2_squared"], "invalid length norm"
        self.cost_specifications.append({
            'type': 'length', 'weight': float(weight), 'norm': norm,
        })

        # Get a symbolic expression for the difference between subsequent
        # control points, in terms of the original decision variables
        control_points = self.bezier_path_u.control_points()

        A = []
        for i in range(self.order):
            with warnings.catch_warnings():
                # ignore numpy warnings about subtracting symbolics
                warnings.simplefilter('ignore', category=RuntimeWarning)
                diff = control_points[i] - control_points[i+1]
            A.append(DecomposeLinearExpressions(diff.flatten(),
                        self.u_vars))
            
        # Apply a cost to the starting segment of each edge. 
        for edge in self.gcs.Edges():
            if edge.u() != self.source:
                x = edge.xu()
                for i in range(self.order):
                    if norm == "L1":
                        cost = L1NormCost(weight*A[i], np.zeros(self.dim))
                    elif norm == "L2":
                        cost = L2NormCost(weight*A[i], np.zeros(self.dim))
                    else:  # L2 squared
                        cost = QuadraticCost(
                                Q=weight*A[i].T@A[i], b=np.zeros(len(x)), c=0.0)

                    edge.AddCost(Binding[Cost](cost, edge.xu()))

    def AddDerivativeCost(self, degree, weight=1.0, norm="L2"):
        """
        Add a penalty on the derivative of the path. We do this by penalizing
        some norm of the control points of the derivative of the path.
        
        Args:
            degree: The derivative to penalize. degree=0 is the original
                    trajectory, degree=1 is the first derivative, etc. 
            weight: Weight for this cost, scalar
            norm:   Norm to use to when evaluating distance between control
                    points (see above)
        """
        assert norm in ["L1", "L2", "L2_squared"], "invalid length norm"
        assert degree >= 0
        assert degree < self.order
        self.cost_specifications.append({
            'type': 'derivative', 'degree': int(degree),
            'weight': float(weight), 'norm': norm,
        })

        # Compute normalization term
        normal = 1
        pro_now = self.order
        for num in range(degree):
            normal *= pro_now
            pro_now -= 1
        
        # Get a symbolic version of the i^th derivative of a segment
        path_deriv = self.bezier_path_u.MakeDerivative(degree)

        # Get a symbolic expression for the difference between subsequent
        # control points in the i^th derivative, in terms of the original
        # decision variables
        deriv_control_points = path_deriv.control_points()

        A = []
        for i in range(self.order - degree + 1):
            with warnings.catch_warnings():
                # ignore numpy warnings about subtracting symbolics
                warnings.simplefilter('ignore', category=RuntimeWarning)
                diff = deriv_control_points[i]# - deriv_control_points[i+1]
            A.append(DecomposeLinearExpressions(diff.flatten(),
                        self.u_vars)/normal)

        # Apply a cost to the starting segment of each edge. 
        for edge in self.gcs.Edges():
            if edge.u() != self.source:
                x = edge.xu()
                for i in range(self.order - degree + 1):
                    if norm == "L1":
                        cost = L1NormCost(weight*A[i], np.zeros(self.dim))
                    elif norm == "L2":
                        cost = L2NormCost(weight*A[i], np.zeros(self.dim))
                    else:  # L2 squared
                        cost = QuadraticCost(
                                Q=(weight*A[i].T@A[i]), b=np.zeros(len(x)), c=0.0)

                    edge.AddCost(Binding[Cost](cost, x))

    def AddTimeDerivativeCost(self, degree=2, weight=1.0, norm="L1"):
        """Penalize variation of the Bézier *time parameterization*.

        This is deliberately distinct from :meth:`AddDerivativeCost`, which
        acts on the spatial curve.  For a segment with normalized Bézier
        parameter ``u``, this method penalizes control points of
        ``d^degree t(u) / du^degree``.  With ``degree=2`` it is a convex
        regularizer for changes in clock rate within a segment; it is *not* a
        physical acceleration or jerk cost for ``x(t)``.

        In particular, L1 retains the all-linear objective used by Puzzle-1.
        The method is useful only when its weight is explicitly requested:
        the paper benchmarks continue to use spatial costs alone.
        """
        assert norm in ["L1", "L2", "L2_squared"], "invalid time derivative norm"
        assert degree >= 1
        assert degree < self.order
        assert weight >= 0
        self.cost_specifications.append({
            'type': 'time_derivative', 'degree': int(degree),
            'weight': float(weight), 'norm': norm,
            'domain': 'normalized_bezier_parameter_u',
            'interpretation': 'time_parameterization_smoothness',
        })

        time_deriv = self.bezier_duration_u.MakeDerivative(degree)
        deriv_control_points = time_deriv.control_points()
        A = []
        for point in deriv_control_points:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                A.append(DecomposeLinearExpressions(np.asarray(point).flatten(), self.u_vars))

        for edge in self.gcs.Edges():
            if edge.u() == self.source:
                continue
            x = edge.xu()
            for matrix in A:
                if norm == "L1":
                    cost = L1NormCost(weight * matrix, np.zeros(1))
                elif norm == "L2":
                    cost = L2NormCost(weight * matrix, np.zeros(1))
                else:
                    cost = QuadraticCost(
                        Q=weight * matrix.T @ matrix,
                        b=np.zeros(len(x)), c=0.0,
                    )
                edge.AddCost(Binding[Cost](cost, x))

    def AddVelocityConstraint(self, lower_bound, upper_bound):
        """
        Add velocity in each edge, only consider box bound
        lower_bound: a list of lower bound, should compatible with dimension
        upper_bound: a list of upper bound, should compatible with dimension
        """
        assert len(lower_bound) == self.dim
        assert len(upper_bound) == self.dim

        u_path_control = self.bezier_path_u.MakeDerivative(1).control_points()
        u_time_control = self.bezier_duration_u.MakeDerivative(1).control_points()
        lb = np.expand_dims(lower_bound, 1)
        ub = np.expand_dims(upper_bound, 1)

        for ii in range(len(u_path_control)):
            A_ctrl = DecomposeLinearExpressions(u_path_control[ii], self.u_vars)
            b_ctrl = DecomposeLinearExpressions(u_time_control[ii], self.u_vars)
            A_constraint = np.vstack((A_ctrl - ub * b_ctrl, -A_ctrl + lb * b_ctrl))
            velocity_con = LinearConstraint(
                A_constraint, -np.inf*np.ones(2*self.dim), np.zeros(2*self.dim))

            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddConstraint(Binding[Constraint](velocity_con, edge.xu()))

    def SetupShortestPathProblem(self, reset_information, time_constraint_information, vertices_time_bound, vertices_activate_var, lower_bound, upper_bound):
        """
        Formulate a shortest path through convex sets problem where the path is
        composed of bezier curves that must be contained in each convex set.

        Returns:
            source: Drake Gcs Vertex corresponding to the initial convex set
            target: Drake Gcs Vertex corresponding to the final convex set
        """
        # Define vertices. The convex sets for each vertex are such that each
        # control point must be contained in the corresponding region
        gcs_verts = {}  # map our vertices to GCS vertices
        gcs_verts_inverse = {} # mep GCS vertices to our vertices

        #hdot_min = 1e-6
        hdot_min = 1e-6
        A_time_duration = np.vstack((np.eye(self.order + 1), -np.eye(self.order + 1),
                            np.eye(self.order, self.order + 1) - np.eye(self.order, self.order + 1, 1)))
        b_time_duration = np.concatenate((self.end_time * np.ones(self.order + 1), np.zeros(self.order + 1), -hdot_min * np.ones(self.order)))
        self.time_scaling_set = HPolyhedron(A_time_duration, b_time_duration)

        A_time_var = np.array([[1.], [-1.]])
        b_time_var = np.array([self.end_time, 0.]) #0.])
        self.time_var_set = HPolyhedron(A_time_var, b_time_var).CartesianPower(len(self.time_var_list))
        new_time_flag = (lower_bound == None)

        if new_time_flag == False:
            u_path_dev = self.bezier_path_u.MakeDerivative(1).control_points()
            u_time = self.bezier_duration_u.control_points()
            u_time_dev = self.bezier_duration_u.MakeDerivative(1).control_points()
            lb = np.expand_dims(lower_bound, 1)
            ub = np.expand_dims(upper_bound, 1)

        for v in self.vertices:
            tmp_vertex = self.regions[v].CartesianPower(self.order + 1)
            if v not in vertices_time_bound.keys(): # for source and target vertex
                gcs_verts[v] = self.gcs.AddVertex(tmp_vertex.CartesianProduct(self.time_scaling_set).CartesianProduct(self.time_var_set))
            else:
                b_time_duration_new = np.concatenate((vertices_time_bound[v]['c0'][1] * np.ones(self.order + 1),
                                                      -vertices_time_bound[v]['c0'][0] * np.ones(self.order + 1),
                                                      -hdot_min * np.ones(self.order)))
                tmp_vertex = tmp_vertex.CartesianProduct(HPolyhedron(A_time_duration, b_time_duration_new))
                for num in range(len(self.time_var_list)):
                    time_var = self.time_var_list[num]
                    b_time_var_new = np.array(
                        [vertices_time_bound[v][time_var][1], -vertices_time_bound[v][time_var][0]])
                    # b_time_new = np.array([vertices_time_bound[v][time_var][1] + 1e-5, -vertices_time_bound[v][time_var][0]+ 1e-5])
                    new_vertex = HPolyhedron(A_time_var, b_time_var_new)
                    tmp_vertex = tmp_vertex.CartesianProduct(new_vertex)

                # print(self.ts_states[v])
                # print(vertices_time_bound[v])
                # print("--------------------------------------------")
                if new_time_flag == False:
                    A = tmp_vertex.A().copy()
                    b = list(tmp_vertex.b())
                    for num in range(len(self.time_var_list)):
                        time_var = self.time_var_list[num]
                        with warnings.catch_warnings():
                            # ignore numpy warnings about subtracting symbolics
                            warnings.simplefilter('ignore', category=RuntimeWarning)
                            exp = u_time[-1] - u_time[0] + self.u_timevar[num]
                        #print(DecomposeLinearExpressions(exp, self.u_vars))
                        A = np.vstack((A, DecomposeLinearExpressions(exp, self.u_vars)))
                        b.append(vertices_time_bound[v][time_var][1])

                    A = np.vstack((A, DecomposeLinearExpressions(u_time[0] - u_time[-1], self.u_vars)))
                    b.append(-self.min_segment_duration)

                    for ii in range(len(u_path_dev)):
                        with warnings.catch_warnings():
                            # ignore numpy warnings about subtracting symbolics
                            warnings.simplefilter('ignore', category=RuntimeWarning)
                            exp_A = u_path_dev[ii]
                            exp_b = u_time_dev[ii]
                        A_ctrl = DecomposeLinearExpressions(exp_A, self.u_vars)
                        b_ctrl = DecomposeLinearExpressions(exp_b, self.u_vars)
                        A_constraint = np.vstack((A_ctrl - ub * b_ctrl, -A_ctrl + lb * b_ctrl))
                        A = np.vstack((A, A_constraint))
                        b = b + [0 for dim_num in range(2 * self.dim)]

                    tmp_vertex = HPolyhedron(A, b)
                    # print(A)
                    # print(b)
                    # print("------------------------")
                gcs_verts[v] = self.gcs.AddVertex(tmp_vertex)

            gcs_verts_inverse[gcs_verts[v]] = v

        # print(A)
        # print(b)
        
        # Define edges
        gcs_edge_to_product_edge = {}
        for e in self.edges:
            # Get vertex IDs of source and target for this edge
            u = gcs_verts[e[0]]
            v = gcs_verts[e[1]]

            gcs_edge = self.gcs.AddEdge(u, v)
            # Preserve the product-edge identity.  Drake exposes only its
            # own edge ids in a solved path; this map makes it possible to
            # audit the actual TA-state transitions later without re-solving.
            gcs_edge_to_product_edge[str(gcs_edge.id())] = e
            #print(e)
       
        # Define source and target vertices
        source = gcs_verts[self.start_vertex]
        target = gcs_verts[self.end_vertex]

        # Add continuity constraints. This includes both continuity (first and
        # last control points line up) and smoothness (first and last control
        # points of derivatives of the path line up).
        for i in range(self.continuity + 1):
            # N.B. i=0 corresponds to the path itsefl
            path_u_deriv = self.bezier_path_u.MakeDerivative(i)
            path_v_deriv = self.bezier_path_v.MakeDerivative(i)

            time_u_deriv = self.bezier_duration_u.MakeDerivative(i)
            time_v_deriv = self.bezier_duration_v.MakeDerivative(i)

            with warnings.catch_warnings():
                # ignore numpy warnings about subtracting symbolics
                warnings.simplefilter('ignore', category=RuntimeWarning)
                continuity_path_err = path_v_deriv.control_points()[0] - \
                                 path_u_deriv.control_points()[-1]
                continuity_time_err = time_v_deriv.control_points()[0] - \
                                 time_u_deriv.control_points()[-1]

            continuity_path_constraint = LinearEqualityConstraint(
                    DecomposeLinearExpressions(
                        continuity_path_err,
                        self.edge_vars),
                    np.zeros(self.dim))

            continuity_time_constraint = LinearEqualityConstraint(
                DecomposeLinearExpressions(
                    continuity_time_err,
                    self.edge_vars),
                np.zeros(1))

            # Apply the continuity constraints to each edge in the graph
            for edge in self.gcs.Edges():
                if edge.v() != target and edge.u() != source:
                    edge.AddConstraint(Binding[Constraint](
                        continuity_path_constraint,
                        np.concatenate((edge.xu(), edge.xv()))))

                    edge.AddConstraint(Binding[Constraint](
                        continuity_time_constraint,
                        np.concatenate((edge.xu(), edge.xv()))))

        # Add time constraint to each edge in the graph
        for edge in self.gcs.Edges():
            left_state = edge.u()
            right_state = edge.v()
            if left_state == source:
                # location constraint
                for i in range(self.dim):
                    edge.AddConstraint(edge.xv()[i] == self.start_point[i])
                # time constraint
                for num in range(len(self.time_var_list)):
                    edge.AddConstraint(edge.xv()[-1 - num] == 0)

                edge.AddConstraint(edge.xv()[self.dim*(self.order+1)] == 0)

            else:
                my_edge = (gcs_verts_inverse[left_state], gcs_verts_inverse[right_state])
                u_time_control = self.bezier_duration_u.control_points()
                reset_index = []
                if my_edge in time_constraint_information.keys():
                    reset_index = [self.time_var_list.index(time_var) for time_var in reset_information[my_edge]]

                #print(my_edge)
                for index in range(len(self.time_var_list)):
                    # print(gcs_verts_inverse[left_state])
                    # print(vertices_activate_var[gcs_verts_inverse[left_state]])
                    time_var = self.time_var_list[index]
                    if index in reset_index:
                        reset_constraint = LinearEqualityConstraint(DecomposeLinearExpressions(
                            np.array([self.u_timevar[index]]), self.u_vars), np.zeros(1))
                        edge.AddConstraint(Binding[Constraint](
                            reset_constraint,
                            edge.xv()))
                    elif time_var in vertices_activate_var[gcs_verts_inverse[left_state]]:

                        if new_time_flag:
                            edge_time_cons = LinearConstraint(DecomposeLinearExpressions(self.u_timevar[index] + u_time_control[-1] - u_time_control[0], self.u_vars),
                                                              -np.inf*np.ones(1),
                                                              vertices_time_bound[gcs_verts_inverse[edge.u()]][
                                                                  time_var][
                                                                  1]*np.ones(1))
                            edge.AddConstraint(Binding[Constraint](
                                edge_time_cons,
                                edge.xu()))

                        time_continuous_cons = LinearEqualityConstraint(
                            DecomposeLinearExpressions(self.u_timevar[index] + u_time_control[-1] - u_time_control[0] -
                                            self.v_timevar[index], self.edge_vars), np.zeros(1))
                        edge.AddConstraint(Binding[Constraint](
                            time_continuous_cons,
                            np.concatenate((edge.xu(), edge.xv()))))

                # if right_state == target:  # add final velocity constraint, only when edge have no time constraint
                #     # print(gcs_verts_inverse[edge.u()], gcs_verts_inverse[edge.v()], "target_edge_construction")
                #     u_path_control = self.dummy_path_u.MakeDerivative(1).control_points()
                #     # print(u_path_control)
                #     with warnings.catch_warnings():
                #         # ignore numpy warnings about subtracting symbolics
                #         warnings.simplefilter('ignore', category=RuntimeWarning)
                #         final_velocity_error = u_path_control[-1]
                #
                #     # print(final_velocity_error)
                #     # print(DecomposeLinearExpressions(final_velocity_error, self.dummy_u_vars), "why!!!")
                #     final_velocity_cons = LinearEqualityConstraint(
                #         DecomposeLinearExpressions(final_velocity_error, self.dummy_u_vars), np.zeros(self.dim)) # 0.01 * np.ones(self.dim))
                #     edge.AddConstraint(Binding[Constraint](
                #         final_velocity_cons,
                #         edge.xu()))

        # Allow access to GCS vertices later
        self.gcs_verts = gcs_verts
        self.gcs_verts_inverse = gcs_verts_inverse
        self.gcs_edge_to_product_edge = gcs_edge_to_product_edge
        return (source, target)

    def SolveShortestPath(self, verbose=True, convex_relaxation=False,
            preprocessing=True, max_rounded_paths=0, solver="mosek"):
        """
        Solve the shortest path problem (self.gcs).

        Args:
            verbose: whether to print solver details to the screen
            convex_relaxation: whether to solve the original MICP or the convex
                               relaxation (+rounding)
            preprocessing: preprocessing step to reduce the size of the graph
            max_rounded_paths: number of distinct paths to compare during
                               rounding for the convex relaxation
            solver: underling solver for the CP/MICP. Must be "mosek" or
                    "gurobi"

        Returns:
            result: a MathematicalProgramResult encoding the solution.
        """
        # Set solver options
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = convex_relaxation
        options.preprocessing = preprocessing
        options.max_rounded_paths = max_rounded_paths
        if solver == "mosek":
            options.solver = MosekSolver()
        elif solver == "gurobi":
            options.solver = GurobiSolver()
        else:
            raise ValueError(f"Unknown solver {solver}")
        #options.solver = ScsSolver()
        # options.solver = CsdpSolver()
        solver_opts = SolverOptions()
        solver_opts.SetOption(CommonSolverOption.kPrintToConsole, verbose)
        options.solver_options = solver_opts
        # Solve the problem
        result = self.gcs.SolveShortestPath(self.source, self.target, options)

        return result

    def formulation_statistics(self):
        """Return implementation-independent, pre-preprocessing GCS counts."""
        variables = set()
        equality_rows = 0
        inequality_rows = 0
        constraint_bindings = 0
        cost_bindings = 0

        def variable_id(variable):
            return int(variable.get_id())

        def visit_bindings(bindings):
            nonlocal equality_rows, inequality_rows, constraint_bindings
            for binding in bindings:
                constraint_bindings += 1
                evaluator = binding.evaluator()
                rows = int(evaluator.num_constraints())
                try:
                    lower = np.asarray(evaluator.lower_bound(), dtype=float)
                    upper = np.asarray(evaluator.upper_bound(), dtype=float)
                    equal = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper)
                    equality_rows += int(np.count_nonzero(equal))
                    inequality_rows += rows - int(np.count_nonzero(equal))
                except (AttributeError, TypeError, ValueError):
                    inequality_rows += rows

        for vertex in self.gcs.Vertices():
            variables.update(variable_id(item) for item in vertex.x())
            visit_bindings(vertex.GetConstraints())
            cost_bindings += len(vertex.GetCosts())
        for edge in self.gcs.Edges():
            variables.add(variable_id(edge.phi()))
            variables.update(variable_id(item) for item in edge.xu())
            variables.update(variable_id(item) for item in edge.xv())
            visit_bindings(edge.GetConstraints())
            cost_bindings += len(edge.GetCosts())
        flow_variables = len(self.gcs.Edges())
        return {
            'scope': 'raw GCS before graph preprocessing and solver transcription',
            'symbolic_variables_total': len(variables),
            'symbolic_continuous_variables': len(variables) - flow_variables,
            'flow_variables': flow_variables,
            'original_binary_variables': flow_variables,
            'constraint_bindings': constraint_bindings,
            'equality_constraint_rows': equality_rows,
            'inequality_constraint_rows': inequality_rows,
            'cost_bindings': cost_bindings,
            'note': 'Convex-set membership and cost epigraph rows are included in solver-transcribed counts.',
        }

    @staticmethod
    def _solver_status(result):
        try:
            return str(result.get_solution_result())
        except Exception:
            return None

    @staticmethod
    def _failure_status(raw_status, elapsed, timeout):
        text = (raw_status or '').lower()
        if (timeout and elapsed >= timeout * 0.999) or 'iterationlimit' in text:
            return 'timeout'
        if 'infeasible' in text:
            return 'infeasible'
        return 'error'

    @staticmethod
    def _relaxation_gap_metrics(relaxation_cost, feasible_cost, flow_tolerance):
        """Compute the paper's certified relaxation-gap bound.

        The returned auxiliary feasible-denominator value is descriptive only;
        it is deliberately not used as an optimality certificate.
        """
        difference = feasible_cost - relaxation_cost
        metrics = {
            'relative_gap_percent': None,
            'relative_gap_definition': (
                '(J_feas - J_relax) / J_relax * 100; '
                'a certified upper bound on relative suboptimality when '
                'J_relax is a valid positive relaxation lower bound'
            ),
            'relative_cost_difference_over_feasible_percent': (
                100.0 * difference / feasible_cost if feasible_cost != 0 else None
            ),
        }
        zero_tolerance = 1e-12 * builtins.max(1.0, abs(feasible_cost))
        if relaxation_cost > zero_tolerance:
            metrics['relative_gap_percent'] = 100.0 * difference / relaxation_cost
        elif abs(relaxation_cost) <= zero_tolerance and abs(difference) <= zero_tolerance:
            metrics['relative_gap_percent'] = 0.0
        else:
            metrics['relative_gap_status'] = (
                'undefined_zero_or_negative_relaxation_lower_bound'
            )
        if relaxation_cost > feasible_cost:
            excess = relaxation_cost - feasible_cost
            threshold = builtins.max(1e-8, flow_tolerance * abs(feasible_cost))
            metrics['numerical_warning'] = (
                'J_relax exceeds J_feas; raw objectives and negative gap are '
                f'preserved (excess={excess:.6g}, '
                f'within_flow_scaled_tolerance={excess <= threshold})'
            )
        return metrics

    def _make_gcs_options(self, *, relaxation, preprocessing, verbose, solver,
                          timeout, log_path=None):
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = relaxation
        options.preprocessing = preprocessing
        options.max_rounded_paths = 0
        if solver == 'mosek':
            options.solver = MosekSolver()
        elif solver == 'gurobi':
            options.solver = GurobiSolver()
        else:
            raise ValueError(f"Unknown solver {solver}")
        solver_opts = SolverOptions()
        solver_opts.SetOption(CommonSolverOption.kPrintToConsole, verbose)
        if log_path is not None:
            solver_opts.SetOption(CommonSolverOption.kPrintFileName, str(log_path))
        if timeout is not None and timeout > 0:
            option_name = ('MSK_DPAR_OPTIMIZER_MAX_TIME' if solver == 'mosek'
                           else 'TimeLimit')
            solver_opts.SetOption(options.solver.id(), option_name, float(timeout))
        options.solver_options = solver_opts
        return options

    def _integral_path(self, result, tolerance):
        edge_values = [(edge, float(result.GetSolution(edge.phi())))
                       for edge in self.gcs.Edges()]
        if any(tolerance < value < 1.0 - tolerance for _, value in edge_values):
            return None
        path = []
        vertex = self.source
        visited = {vertex.id()}
        while vertex != self.target:
            outgoing = [(edge, value) for edge, value in edge_values
                        if edge.u() == vertex and value >= 1.0 - tolerance]
            if len(outgoing) != 1:
                return None
            edge = outgoing[0][0]
            path.append(edge)
            vertex = edge.v()
            if vertex.id() in visited:
                return None
            visited.add(vertex.id())
        return path

    def _sample_path(self, result, rng, tolerance):
        path = []
        vertex = self.source
        visited = {vertex.id()}
        for _ in range(len(self.gcs.Vertices()) + 1):
            if vertex == self.target:
                return path
            candidates, weights = [], []
            for edge in vertex.outgoing_edges():
                if edge.v().id() in visited:
                    continue
                value = builtins.max(0.0, float(result.GetSolution(edge.phi())))
                if value > tolerance:
                    candidates.append(edge)
                    weights.append(value)
            if not candidates:
                return None
            probabilities = np.asarray(weights) / np.sum(weights)
            edge = candidates[int(rng.choice(len(candidates), p=probabilities))]
            path.append(edge)
            vertex = edge.v()
            visited.add(vertex.id())
        return None

    def SolveShortestPathWithReport(
            self, verbose=False, preprocessing=True, max_rounded_paths=10,
            max_rounding_trials=100, flow_tolerance=1e-5, rounding_seed=0,
            timeout=7200.0, solver='mosek', log_dir=None):
        """Solve and expose relaxation/rounding diagnostics.

        Rounding samples unique paths from the relaxed edge flow and solves a
        convex restriction for each.  ``rounding_attempts`` is precisely the
        number of convex restrictions submitted to a solver; sampling retries
        and duplicate paths are reported separately.
        """
        started = time.perf_counter()
        log_dir = Path(log_dir) if log_dir is not None else None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
        relax_log = log_dir / 'relaxation.log' if log_dir is not None else None
        options = self._make_gcs_options(
            relaxation=True, preprocessing=preprocessing, verbose=verbose,
            solver=solver, timeout=timeout, log_path=relax_log)
        relaxation = self.gcs.SolveShortestPath(self.source, self.target, options)
        elapsed_relaxation = time.perf_counter() - started
        diagnostics = {
            'method': 'auditable_flow_sampling_and_convex_restriction',
            'solver': solver,
            'preprocessing': preprocessing,
            'rounding_seed': rounding_seed,
            'flow_tolerance': flow_tolerance,
            'max_rounding_trials': max_rounding_trials,
            'max_rounded_paths': max_rounded_paths,
            'timeout_sec': timeout,
            'solver_tolerances': solver_tolerances(solver),
            'relaxation_status_raw': self._solver_status(relaxation),
            'relaxation_time_sec': elapsed_relaxation,
            'j_relax': float(relaxation.get_optimal_cost()) if relaxation.is_success() else None,
            'j_feas': None,
            'relative_gap_percent': None,
            'relative_gap_definition': (
                '(J_feas - J_relax) / J_relax * 100; '
                'a certified upper bound on relative suboptimality when '
                'J_relax is a valid positive relaxation lower bound'
            ),
            # This was the original reporting quantity.  Retain it for
            # backwards-compatible descriptive analysis, but do not call it
            # an optimality gap: it is not an upper bound on suboptimality.
            'relative_cost_difference_over_feasible_percent': None,
            'tight': False,
            'sampling_trials': 0,
            'duplicate_paths': 0,
            'rounding_attempts': 0,
            'rounding_success': False,
            'attempts': [],
            'status': 'error',
            'solver_transcription_relaxation': parse_mosek_log(relax_log)
                if solver == 'mosek' and relax_log is not None else None,
        }
        if not relaxation.is_success():
            diagnostics['solve_time_sec'] = time.perf_counter() - started
            diagnostics['status'] = self._failure_status(
                diagnostics['relaxation_status_raw'], diagnostics['solve_time_sec'], timeout)
            return relaxation, diagnostics

        integral_path = self._integral_path(relaxation, flow_tolerance)
        if integral_path is not None:
            diagnostics.update({
                'j_feas': diagnostics['j_relax'],
                'relative_gap_percent': 0.0,
                'tight': True,
                'rounding_success': True,
                'status': 'feasible',
                'active_edge_ids': [str(edge.id()) for edge in integral_path],
                'solve_time_sec': time.perf_counter() - started,
            })
            return relaxation, diagnostics

        rng = np.random.default_rng(rounding_seed)
        unique_paths = set()
        best_result = None
        best_cost = math.inf
        for trial in range(max_rounding_trials):
            if len(unique_paths) >= max_rounded_paths:
                break
            if timeout is not None and time.perf_counter() - started >= timeout:
                diagnostics['status'] = 'timeout'
                break
            diagnostics['sampling_trials'] += 1
            path = self._sample_path(relaxation, rng, flow_tolerance)
            if not path:
                continue
            key = tuple(str(edge.id()) for edge in path)
            if key in unique_paths:
                diagnostics['duplicate_paths'] += 1
                continue
            unique_paths.add(key)
            remaining = None if timeout is None else builtins.max(1e-6, timeout - (time.perf_counter() - started))
            attempt_log = (log_dir / f'restriction-{len(unique_paths):03d}.log'
                           if log_dir is not None else None)
            restriction_options = self._make_gcs_options(
                relaxation=False, preprocessing=False, verbose=verbose,
                solver=solver, timeout=remaining, log_path=attempt_log)
            attempt_started = time.perf_counter()
            candidate = self.gcs.SolveConvexRestriction(path, restriction_options)
            attempt_time = time.perf_counter() - attempt_started
            diagnostics['rounding_attempts'] += 1
            attempt = {
                'index': diagnostics['rounding_attempts'],
                'edge_ids': list(key),
                'status_raw': self._solver_status(candidate),
                'success': bool(candidate.is_success()),
                'time_sec': attempt_time,
                'objective': float(candidate.get_optimal_cost()) if candidate.is_success() else None,
                'solver_transcription': parse_mosek_log(attempt_log)
                    if solver == 'mosek' and attempt_log is not None else None,
            }
            diagnostics['attempts'].append(attempt)
            if candidate.is_success() and attempt['objective'] < best_cost:
                best_result, best_cost = candidate, attempt['objective']

        diagnostics['solve_time_sec'] = time.perf_counter() - started
        if best_result is not None:
            diagnostics['rounding_success'] = True
            diagnostics['status'] = 'feasible'
            diagnostics['j_feas'] = best_cost
            diagnostics['active_edge_ids'] = diagnostics['attempts'][
                builtins.min(range(len(diagnostics['attempts'])),
                    key=lambda i: diagnostics['attempts'][i]['objective']
                    if diagnostics['attempts'][i]['objective'] is not None else math.inf)
            ]['edge_ids']
            # Match Marcucci et al., "Motion Planning around Obstacles with
            # Convex Optimization": C_relax <= C_opt <= C_round implies
            # (C_round - C_opt) / C_opt <=
            # (C_round - C_relax) / C_relax.  The latter is therefore the
            # reported certified relaxation optimality-gap bound.
            diagnostics.update(self._relaxation_gap_metrics(
                diagnostics['j_relax'], best_cost, flow_tolerance
            ))
            return best_result, diagnostics
        if timeout and diagnostics['solve_time_sec'] >= timeout * 0.999:
            diagnostics['status'] = 'timeout'
        else:
            diagnostics['recovery_status'] = 'rounding_failed'
        return relaxation, diagnostics
    
    def PlotScenario(self):
        """
        Add a plot of each region to the current matplotlib axes. 
        Only supports 2D polytopes for now.
        """
        for vertex, region in self.regions.items():
            assert region.ambient_dimension() == 2, "only 2D sets allowed"

            if vertex != self.end_vertex:
                # The target vertex is trivial and therefore not plotted

                # Compute vertices of the polygon in known order
                v = VPolytope(region).vertices().T
                hull = ConvexHull(v)
                v_sorted = np.vstack([v[hull.vertices,0],v[hull.vertices,1]]).T

                # Make a polygonal patch
                poly = Polygon(v_sorted, alpha=0.5, edgecolor="k", linewidth=3)
                plt.gca().add_patch(poly)
       
        # Use equal axes so square things look square
        plt.axis('equal')

    def get_outgoing_edge(self, result, vertex):
        # Helper function that returns the highest probability (phi-value)
        # edge leading out of the given vertex
        edge = None
        phi = -1.0
        for e in self.gcs.Edges():
            if (e.u() == vertex) and result.GetSolution(e.phi()) > phi:
                edge = e
                phi = result.GetSolution(e.phi())
        return edge

    def ExtractSolution(self, result):
        """
        Extract a sequence of bezier curves representing the optimal solution.

        Args:
            result: MathematicalProgramResult from calling SolveShortestPath

        Returns:
            A list of BsplineTrajectory objects representing each segement of
            the optimal solution.
        """
        # List of bezier curves that we'll return
        curves = []

        # Traverse the graph along the optimal path
        v_tmp = self.gcs_verts[self.start_vertex]
        e = self.get_outgoing_edge(result, v_tmp)
        v = e.v()
        while v != self.gcs_verts[self.end_vertex]:
            e = self.get_outgoing_edge(result, v)
           
            # Get the control points for this vertex
            xu = result.GetSolution(e.xu())
            print("reach ", self.ts_states[self.gcs_verts_inverse[e.u()]], " at ", xu[self.dim*(self.order+1)])
            print("the duration time is ", xu[self.dim*(self.order+1) + self.order] - xu[self.dim*(self.order+1)])
            print([xu[-len(self.time_var_list)+i] for i in range(len(self.time_var_list))])
            print(self.time_var_list)
            print(xu)
            print("-----------------------------------")
            control_points = xu[0:self.dim*(self.order+1)].reshape(self.order+1, -1)
            basis = BsplineBasis(self.order+1, self.order+1, 
                                 KnotVectorType.kClampedUniform, 0, 1)
            curves.append(BsplineTrajectory(basis, control_points.T))

            # move to the next vertex
            v = e.v()

        value = result.GetSolution(v.x())
        print("reach target at ", value[self.dim*(self.order+1)])

        return curves
    
    def AnimateSolution(self, result, show=True, save=False, filename=None):
        """
        Create an animation of the solution on the current set of matplotlib
        axes.

        Args:
            result:   MathematicalProgramResult from calling SolveShortestPath
            show:     Flag for displaying the animation immediately
            save:     Flag for saving a gif of the animation
            filename: String denoting what file to save the animation to (.gif)
        """
        assert self.dim == 2, "animation only supported in 2D"

        # Get current matplotlib figure and axes
        fig = plt.gcf()
        ax = plt.gca()

        # Get the solution as a sequence of splines
        s = self.ExtractSolution(result)
        q = ax.scatter(*s[0].value(0), color='blue', s=50, zorder=3)

        def animate(t):
            segment = floor(t)
            new_q = s[segment].value(t % 1).T
            q.set_offsets(new_q)
            return q

        t = np.arange(0, len(s), 0.02)
        ani = animation.FuncAnimation(fig, animate, t, interval=50, blit=False)

        if save:
            assert filename is not None, "must supply a filename to save the animation"
            print(f"Saving animation to {filename}, this may take a minute...")
            ani.save(filename, writer=animation.PillowWriter(fps=30))

        if show:
            plt.show()

        return ani

    def PlotSolution(self, result, plot_control_points=True, plot_path=True):
        """
        Add a plot of the solution to the current matplotlib axes. Only
        supported for 2D. 

        Args:
            result: MathematicalProgramResult from calling SolveShortestPath
            plot_control_points: flag for plotting the control points
            plot_path: flag for plotting the actual path
        """

        for edge in self.gcs.Edges():
            # Note that focusing on xu for each edge ignores the target
            # state, since that's just an indicator of task completion
            if edge.u() != self.source:
                phi = result.GetSolution(edge.phi())
                #value = []
                #for num in range(2*(self.order+1)):
                 #   value.append(result.GetSolution(edge.xu()[num]))
                #xu = np.array(value)
                #print("---------------------")
                #print(phi, "test!!!!!!!!!")
                #xu = result.GetSolution(edge.xu())

                if phi > 0.0:
                    # Construct a bezier curve from the control points
                    xu = result.GetSolution(edge.xu())
                    #print(self.gcs_verts_inverse[edge.u()], self.gcs_verts_inverse[edge.v()])
                    control_points = xu[0:2*(self.order+1)].reshape(self.order+1, -1)
                    basis = BsplineBasis(self.order+1, self.order+1,
                                         KnotVectorType.kClampedUniform, 0, 1)
                    path = BsplineTrajectory(basis, control_points.T)

                    if plot_control_points:
                        plt.plot(control_points[:,0], control_points[:,1], 'o--',
                                color='red', alpha=phi)

                    if plot_path:
                        curve = path.vector_values(np.linspace(0,1))
                        plt.plot(curve[0,:], curve[1,:], color='blue', linewidth=3,
                                alpha=phi)


    def get_trajectory_and_time(self, result):
        # Extract trajectory control points
        knots = np.zeros(self.order + 1)
        path_control_points = []
        time_control_points = []

        v = self.gcs_verts[self.start_vertex]
        while v != self.gcs_verts[self.end_vertex]:
            edge = self.get_outgoing_edge(result, v)
            if edge.v() != self.gcs_verts[self.end_vertex]:
                edge_time = knots[-1] + 1.
                knots = np.concatenate((knots, np.full(self.order, edge_time)))
                if len(self.time_var_list) == 0:
                    edge_path_points = np.reshape(result.GetSolution(edge.xv())[:-(self.order + 1)],
                                                  (self.dim, self.order + 1), "F")
                    edge_time_points = result.GetSolution(edge.xv())[-(self.order + 1):]
                else:
                    edge_path_points = np.reshape(result.GetSolution(edge.xv())[:-(self.order + 1 + len(self.time_var_list))],
                                                  (self.dim, self.order + 1), "F")
                    edge_time_points = result.GetSolution(edge.xv())[-(self.order + 1 + len(self.time_var_list)): - len(self.time_var_list)]
                for ii in range(self.order):
                    path_control_points.append(edge_path_points[:, ii])
                    time_control_points.append(np.array([edge_time_points[ii]]))

            # move to the next vertex
            v = edge.v()

        offset = time_control_points[0].copy()
        for ii in range(len(time_control_points)):
            time_control_points[ii] -= offset

        path_control_points = np.array(path_control_points).T
        time_control_points = np.array(time_control_points).T

        path = BsplineTrajectory(BsplineBasis(self.order + 1, knots), path_control_points)
        time_traj = BsplineTrajectory(BsplineBasis(self.order + 1, knots), time_control_points)

        return BezierTrajectory(path, time_traj)

    def _ta_state_components(self, state):
        """Flatten a composed TA node into template-local state descriptors."""
        metadata = {
            int(item['template_index']): str(item['operator'])
            for item in getattr(self, 'ta_template_metadata', [])
            if item.get('template_index') is not None
        }
        output = {}

        def visit(value, path):
            # ``res`` / unfolded GF nodes use ("template", state, copy),
            # whereas ordinary templates retain their integer global state.
            if (isinstance(value, tuple) and len(value) == 3 and
                    isinstance(value[0], str) and
                    isinstance(value[1], (int, np.integer))):
                template = int(value[0])
                raw_state = int(value[1])
                local_state = raw_state - 5 * template if raw_state >= 5 * template else raw_state
                operator = metadata.get(template)
                # The implementation stores GF states in graph-construction
                # order 0 -> 4 -> 1 -> 2.  This differs from the paper's
                # semantic names s0 -> s1 -> s2 -> s4, so never expose the
                # implementation number as if it were the paper state.
                paper_index = ({0: 0, 4: 1, 1: 2, 3: 3, 2: 4}.get(local_state, local_state)
                               if operator == 'GF' else local_state)
                component = {
                    'template_index': template,
                    'operator': operator,
                    'state_index': local_state,
                    'unfold_copy': int(value[2]) if isinstance(value[2], (int, np.integer)) else value[2],
                    'paper_state': f's_{paper_index}',
                }
                # The manually constructed Rover ``res`` TA follows the
                # Appendix-B map exactly.  Preserve its *region-map name*,
                # rather than forcing a video consumer to reinterpret a
                # private node number: states 2 in copies 0/1 are R_0/R_1,
                # and state 4 is R_4.
                if operator == 'res':
                    copy = component['unfold_copy']
                    if local_state == 5:
                        component['paper_state'] = 's_10'
                    elif isinstance(copy, int):
                        component['paper_state'] = f's_{local_state + 5 * copy}'
                    component['region_map'] = {
                        0: 'R_3', 1: 'R_2',
                        2: 'R_0' if component['unfold_copy'] == 0 else 'R_1',
                        3: 'universe', 4: 'R_4', 5: 'universe',
                    }.get(local_state)
                output[path] = component
            elif isinstance(value, (int, np.integer)):
                global_state = int(value)
                template = global_state // 5
                operator = metadata.get(template)
                local_state = global_state - 5 * template
                paper_index = ({0: 0, 4: 1, 1: 2, 3: 3, 2: 4}.get(local_state, local_state)
                               if operator == 'GF' else local_state)
                output[path] = {
                    'template_index': template,
                    'operator': operator,
                    'state_index': local_state,
                    'unfold_copy': None,
                    'paper_state': f's_{paper_index}',
                }
            elif isinstance(value, tuple):
                for index, child in enumerate(value):
                    visit(child, path + (index,))

        visit(state, ())
        return output

    def get_ta_key_state_schedule(self, result, trajectory=None):
        """Return physical-time transitions of the solved product TA path.

        The method does not infer states from a sampled trajectory: it follows
        the positive-flow GCS path and uses the edge-to-product map created in
        :meth:`SetupShortestPathProblem`.  ``time_sec`` is the start of the
        destination product segment, i.e. the instant at which the product
        enters its new TA state.  Dummy source/target endpoints are retained
        only as metadata and never emitted as a state-change event.
        """
        edge_map = getattr(self, 'gcs_edge_to_product_edge', {})
        if not edge_map or not hasattr(self, 'ts_states'):
            return []
        if trajectory is None:
            trajectory = self.get_trajectory_and_time(result)
        try:
            boundaries = np.asarray(trajectory.get_segment_times(), dtype=float)
        except (AttributeError, RuntimeError, TypeError):
            boundaries = np.asarray([], dtype=float)

        events = []
        vertex = self.gcs_verts[self.start_vertex]
        segment_index = 0
        while vertex != self.gcs_verts[self.end_vertex]:
            edge = self.get_outgoing_edge(result, vertex)
            if edge is None:
                break
            product_edge = edge_map.get(str(edge.id()))
            if product_edge is not None:
                product_u, product_v = product_edge
                state_u = self.ts_states.get(product_u)
                state_v = self.ts_states.get(product_v)
                ta_u = state_u[1] if state_u is not None and len(state_u) > 1 else None
                ta_v = state_v[1] if state_v is not None and len(state_v) > 1 else None
                # The source dummy has no TA state.  Treat source -> first
                # product vertex as a t=0 *entry* so templates initially in
                # s1 (G with a=0) or s4 (GF with a+c=0) are not omitted from
                # the video progress record.
                if ta_v is not None and ta_u != ta_v:
                    if len(boundaries):
                        instant = float(boundaries[builtins.min(segment_index, len(boundaries) - 1)])
                    else:
                        instant = None
                    before = self._ta_state_components(ta_u) if ta_u is not None else {}
                    after = self._ta_state_components(ta_v)
                    component_changes = []
                    for component_path in sorted(set(before) | set(after)):
                        source_component = before.get(component_path)
                        target_component = after.get(component_path)
                        if source_component != target_component:
                            component_changes.append({
                                'component_path': list(component_path),
                                'from': source_component,
                                'to': target_component,
                            })
                    label_map = getattr(self, 'ts_label_map', {})
                    events.append({
                        'time_sec': instant,
                        'from_ta_state': repr(ta_u),
                        'to_ta_state': repr(ta_v),
                        'from_product_vertex': repr(product_u),
                        'to_product_vertex': repr(product_v),
                        'gcs_edge_id': str(edge.id()),
                        'segment_index': int(segment_index),
                        'from_region_labels': sorted(label_map.get(state_u[0], set())) if state_u is not None else [],
                        'to_region_labels': sorted(label_map.get(state_v[0], set())),
                        'ta_component_changes': component_changes,
                    })
            if edge.v() != self.gcs_verts[self.end_vertex]:
                segment_index += 1
            vertex = edge.v()
        return events

    def get_switch_schedule(self, result):
        """Return ``(start, end, toggle)`` for active trajectory segments.

        ``switch_vertices`` is populated by the TA/TS product when a state has
        both physical time and an auxiliary dwell clock active.  The method is
        kept separate from ExtractSolution so reporting/headless runs do not
        need its verbose plotting-oriented output.
        """
        schedule = []
        switch_vertices = getattr(self, 'switch_vertices', set())
        vertex = self.gcs_verts[self.start_vertex]
        edge = self.get_outgoing_edge(result, vertex)
        vertex = edge.v()
        while vertex != self.gcs_verts[self.end_vertex]:
            edge = self.get_outgoing_edge(result, vertex)
            xu = result.GetSolution(edge.xu())
            time_index = self.dim * (self.order + 1)
            schedule.append((
                float(xu[time_index]),
                float(xu[time_index + self.order]),
                int(self.gcs_verts_inverse[edge.u()] in switch_vertices),
            ))
            vertex = edge.v()
        return schedule

    def get_trajectory_points(self, result, seg_number):
        # This helper samples uniformly in the spline parameter s and therefore
        # only captures the geometric path shape. It does not sample uniformly
        # in physical time. For time-correct trajectory evaluation, use
        # get_trajectory_and_time() together with BezierTrajectory.value() or
        # BezierTrajectory.vector_values().
        trajectory = self.get_trajectory_and_time(result)
        save_trajectory = np.zeros((seg_number, self.dim))
        #sample_points = np.linspace(0, end_time, seg_number)
        sample_points = np.linspace(trajectory.start_s, trajectory.end_s, seg_number)
        for num in range(sample_points.shape[0]):
            #now_point = trajectory.value(sample_points[num])
            now_point = trajectory.path_traj.value(sample_points[num])
            for dim_num in range(self.dim):
                save_trajectory[num][dim_num] = now_point[dim_num]#[0]

        return save_trajectory

class BezierTrajectory:
    def __init__(self, path_traj, time_traj):
        assert path_traj.start_time() == time_traj.start_time()
        assert path_traj.end_time() == time_traj.end_time()
        self.path_traj = path_traj
        self.time_traj = time_traj
        self.start_s = path_traj.start_time()
        self.end_s = path_traj.end_time()

    def invert_time_traj(self, t):
        if t <= self.start_time():
            return self.start_s
        if t >= self.end_time():
            return self.end_s
        error = lambda s: self.time_traj.value(s)[0, 0] - t
        res = root_scalar(error, bracket=[self.start_s, self.end_s])
        return np.min([np.max([res.root, self.start_s]), self.end_s])

    def value(self, t):
        return self.path_traj.value(self.invert_time_traj(np.squeeze(t)))

    def vector_values(self, times):
        s = [self.invert_time_traj(t) for t in np.squeeze(times)]
        return self.path_traj.vector_values(s)

    def EvalDerivative(self, t, derivative_order=1):
        if derivative_order == 0:
            return self.value(t)
        elif derivative_order == 1:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            r_dot = self.path_traj.EvalDerivative(s, 1)
            return r_dot * s_dot
        elif derivative_order == 2:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            h_ddot = self.time_traj.EvalDerivative(s, 2)[0, 0]
            s_ddot = -h_ddot*(s_dot**3)
            r_dot = self.path_traj.EvalDerivative(s, 1)
            r_ddot = self.path_traj.EvalDerivative(s, 2)
            return r_ddot * s_dot * s_dot + r_dot * s_ddot
        else:
            raise ValueError()


    def start_time(self):
        return self.time_traj.value(self.start_s)[0, 0]

    def end_time(self):
        return self.time_traj.value(self.end_s)[0, 0]

    def get_segment_times(self):
        """Return physical times at every B-spline segment boundary."""
        parameter_times = np.unique(np.asarray(
            self.path_traj.basis().knots(), dtype=float
        ))
        return np.asarray([
            float(self.time_traj.value(value)[0, 0]) for value in parameter_times
        ])

    def rows(self):
        return self.path_traj.rows()

    def cols(self):
        return self.path_traj.cols()
