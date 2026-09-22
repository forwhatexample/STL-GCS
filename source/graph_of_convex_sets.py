import networkx as nx
import numpy as np
from bezier_gcs import BezierGraphOfConvexSets
from pydrake.all import *
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from Time_Automaton import Time_Automaton
import matplotlib.pyplot as plt

class Transition_system_of_convex_sets(nx.DiGraph):

    def __init__(self, dimension=3):
        """
        Create a digraph such that each vertex represents a convex set in configuration space.

        Args:
            node_set: the list of node
            edge_set: the list of edge
            region_map: map each node to a convex set
            label_map: record the label each convex set satisfy
        """

        super().__init__()
        self.region_map = dict()
        self.label_map = dict()
        self.dimension = dimension
        self.node_index = 0
        self.feasible_list = []
        #self.region_of_each_label = dict()

    def add_convex_set(self, convex_set, label):
        """
        Add a new state with the given convex set and labels.
        Note that by default, this state will be disconnected from all
        other states, use AddEdge to add transitions between adjacent or
        overlapping states.

        Args:
            convex_set: a Drake ConvexSet corresponding to this partition
            label: a list of set, each set contains strings representing
                   the predicates that hold in this convex set

        Returns:
            vertex_idx: an integer index representing this vertex
        """
        assert isinstance(convex_set, ConvexSet)
        assert isinstance(label, set)
        assert convex_set.ambient_dimension() == self.dimension

        if isinstance(convex_set, VPolytope):
            convex_set = HPolyhedron(convex_set)
        elif not isinstance(convex_set, HPolyhedron):
            raise TypeError("Only convex polyhedra are currently supported")

        node_name = self.node_index
        self.node_index += 1
        self.add_node(node_name)
        self.region_map[node_name] = convex_set
        self.label_map[node_name] = label

        return node_name

    def AddEdgesFromIntersections(self):
        """
        Add transitions between all partitions that have some non-empty
        intersection.
        """
        for v1 in self.nodes:
            for v2 in self.nodes:
                r1 = self.region_map[v1]
                r2 = self.region_map[v2]
                if r1.IntersectsWith(r2) and (v1 != v2):
                    # print("-----------")
                    # # print(v1)
                    # # print(r1.A())
                    # # print(r1.b())
                    # # print(v2)
                    # # print(r2.A())
                    # # print(r2.b())
                    # print(v1, v2, " have an edge")
                    # print("----------------")
                    self.add_edge(v1, v2)

    def feasible_list_and_label_region_construction(self):
        """
        Compute feasible_list to construct time automaton
        Compute the regions for each label, used in product of GCS and TA
        """
        for node in self.nodes:
            label = self.label_map[node]
            if label not in self.feasible_list:
                self.feasible_list.append(label)
                #self.region_of_each_label[list(label)] = [node]
            #else:
                #self.region_of_each_label[list(label)].append(node)
        return self.feasible_list


    def Product(self, time_automaton, start_point, order, continuity, lower_bound = None, upper_bound = None,
                min_segment_duration=0.25):

        """
               Compute the product of a convex set transition system
                and a Time Automaton (TA), which is a Bezier curve graph of convex sets.

               Args:
                   tima_automaton: Time Automaton corresponding to some STL
                        specification.
                   start_point: starting point, i.e., initial system state
                   order: degree of the bezier curve that will be planned through the
                          graph of convex sets
                   continuity: continuity of the bezier curve that will be planned
                          through the graph of convex sets
                   min_segment_duration: lower bound on the physical duration of
                          each Bezier segment when velocity bounds are enabled

               Returns:
                   gcs: BezierGraphOfConvexSets such that any path through the graph
                        corresponds to a path through transition system that satisfies
                        the STL specification defined by the given TA.
               """
        assert isinstance(time_automaton, Time_Automaton)

        s0 = None
        for s in self.nodes:
            if self.region_map[s].PointInSet(start_point):
                s0 = s

        assert s0 is not None, \
                "the given start point is not contained in any partition"
        
        # Construct vertices: one for each pair of vertices in the TA and transition system
        states = {}
        state_to_vertex = {}
        state_idx = 0
        vertices = []
        vertices_time_bound = dict()
        vertices_activate_var = dict()
        switch_vertices = []
        for s in self.nodes:
            for q in time_automaton.nodes:
                if time_automaton.vertices_to_regions_label[q] <= self.label_map[s]:  #L(q) \subseteq L(s)
                    states[state_idx] = (s, q)
                    state_to_vertex[(s, q)] = state_idx
                    vertices.append(state_idx)
                    vertices_activate_var[state_idx] = time_automaton.activate_time_var[q]
                    # The recovered UR-3 controller toggles its gripper when
                    # entering a state that propagates physical time plus an
                    # auxiliary dwell clock.
                    if len(vertices_activate_var[state_idx]) > 1:
                        switch_vertices.append(state_idx)
                    if q in time_automaton.state_time_bound.keys():
                        vertices_time_bound[state_idx] = time_automaton.state_time_bound[q]
                    state_idx += 1

        # Define regions (convex sets) for each vertex in the graph of convex sets
        regions = {}
        for v in vertices:
            s, q = states[v]
            regions[v] = self.region_map[s]

        # Define the starting vertex
        start_vertex = []
        for v in vertices:
            s, q = states[v]
            if (s == s0) and (q in time_automaton.initial_vertices):
                start_vertex.append(v)
        assert len(start_vertex) != 0, "could not find a valid start vertex"
        # Define edges in the graph of convex sets. An edge between (s,q) and
        # (s',q') exists if L(s) \in L(q), L(s') \in L(q')
        #  and one of the following conditions hold:
        #
        #   1. q = q' and s-->s' in this transition system
        #   2. q-->q' in the TA and s -->s' in this transition system
        #   3. q-->q' in the TA and s = s'

        edges = []
        edge_set = set()
        inner_edges = set()
        outer_edges = set()
        reset_information = {}
        time_constraint_information = {}
        ts_successors = {s: list(self.successors(s)) for s in self.nodes}
        ta_successors = {q: list(time_automaton.successors(q)) for q in time_automaton.nodes}

        def add_product_edge(u_vertex, v_vertex, q_edge=None):
            if u_vertex == v_vertex:
                return
            edge = (u_vertex, v_vertex)
            if edge in edge_set:
                return
            edge_set.add(edge)
            edges.append(edge)
            if q_edge is not None:
                outer_edges.add(edge)
                reset_information[edge] = time_automaton.edges_reset_variables[q_edge]
                time_constraint_information[edge] = time_automaton.edges_time_constraint[q_edge]
            else:
                inner_edges.add(edge)

        for v in vertices:
            s, q = states[v]

            # Inner TS transitions: keep the TA state fixed and move only along
            # existing TS edges.
            if q not in time_automaton.accepting_vertices:
                for s_prime in ts_successors[s]:
                    v_prime = state_to_vertex.get((s_prime, q))
                    if v_prime is not None:
                        add_product_edge(v, v_prime)

            # Outer TA transitions: move in the TA and optionally also move to
            # an adjacent TS region. The corresponding reset/time constraints
            # are copied from the TA edge.
            for q_prime in ta_successors[q]:
                ta_edge = (q, q_prime)

                v_prime = state_to_vertex.get((s, q_prime))
                if v_prime is not None:
                    add_product_edge(v, v_prime, ta_edge)

                for s_prime in ts_successors[s]:
                    v_prime = state_to_vertex.get((s_prime, q_prime))
                    if v_prime is not None:
                        add_product_edge(v, v_prime, ta_edge)


        # Define the ending vertex. There are edges from (s,q) to the end
        # vertex whenever q is in the accepting set of the TA.
        end_vertex = state_idx
        state_idx += 1
        target_dummy_edges = []
        for v in vertices:
            s, q = states[v]
            if q in time_automaton.accepting_vertices:
                edges.append((v, end_vertex))
                target_dummy_edges.append((v, end_vertex))
        vertices.append(end_vertex)
        work_space_low = -100. * np.ones(self.dimension)
        work_space_up = -100. * np.ones(self.dimension)
        zero = np.zeros(self.dimension)
        #regions[end_vertex] = HPolyhedron.MakeBox(work_space_low, work_space_up)
        regions[end_vertex] = HPolyhedron.MakeBox(zero, zero)

        new_start_vertex = state_idx
        #regions[new_start_vertex] = HPolyhedron.MakeBox(work_space_low, work_space_up)
        regions[new_start_vertex] = HPolyhedron.MakeBox(zero, zero)

        source_dummy_edges = []
        for v in start_vertex:
            edges.append((new_start_vertex, v))
            source_dummy_edges.append((new_start_vertex, v))
        vertices.append(new_start_vertex)
        start_vertex = new_start_vertex

        # Construct and return the graph of convex sets
        bgcs = BezierGraphOfConvexSets(vertices, edges, regions, start_vertex, end_vertex, start_point,
                            reset_information, time_constraint_information, time_automaton.end_time, vertices_time_bound,
                            states, vertices_activate_var, order, continuity, lower_bound, upper_bound,
                            min_segment_duration=min_segment_duration)
        bgcs.product_statistics = {
            **time_automaton.reporting_summary(),
            'decomposition_regions': len(self.nodes),
            'decomposition_edges': len(self.edges),
            'gcs_vertices': len(vertices),
            'gcs_edges': len(edges),
            'gcs_inner_edges': len(inner_edges),
            'gcs_outer_edges': len(outer_edges),
            'gcs_source_dummy_edges': len(source_dummy_edges),
            'gcs_target_dummy_edges': len(target_dummy_edges),
            'bezier_degree': order,
            'smoothness_order': continuity,
            'configuration_dimension': self.dimension,
            'min_segment_duration': min_segment_duration,
        }
        # Retain the small, declarative TA/TS maps needed to serialize actual
        # product-path state transitions for reports and videos.  They do not
        # participate in the optimization problem.
        bgcs.ta_template_metadata = [dict(item) for item in getattr(
            time_automaton, 'template_metadata', []
        )]
        bgcs.ts_label_map = {
            node: set(labels) for node, labels in self.label_map.items()
        }
        bgcs.switch_vertices = set(switch_vertices)
        return bgcs

    def visualize(self, color_dict={}, inverse_color_dict={}, inverse_label_dict={}, background='black', edgecolor='black',
                  edgewidth=1.0, alpha=1.0):
        """
        Make a pyplot visualization of the regions on the current pyplot axes.
        Only supports 2D polytopes for now.

        Args:
            color_dict: dictionary mapping color values to partition labels
                        (list of predicates) that take the given color. If a
                        partition does not have a color in this dictionary, it
                        takes a default blue value.
            background: background color
            edgecolor: edge color for each partition
            edgewidth: width of the edge of each partition
            alpha: opacity for each partition
        """
        for vertex in self.nodes:

            region = self.region_map[vertex]
            label = self.label_map[vertex]
            print_label = set([])
            assert region.ambient_dimension() == 2, "only 2D sets allowed"

            # Compute vertices of the polygon in known order
            v = VPolytope(region).vertices().T
            hull = ConvexHull(v)
            v_sorted = np.vstack([v[hull.vertices, 0], v[hull.vertices, 1]]).T

            # Make a polygonal patch
            color = 'white'
            # print(region.A())
            # print(region.b())
            # print(label)
            for c, labels in color_dict.items():
                for col_label in labels:
                    if col_label[0] in label:
                        color = c
                        print_label = set(col_label)

            for c, labels in inverse_color_dict.items():
                for col_label in labels:
                    labels_set = set(col_label)
                    if len(labels_set & label)==0:
                        color = c
                    # print(label, " color is ", c)
            poly = Polygon(v_sorted, alpha=alpha, facecolor=color,
                           edgecolor=edgecolor, linewidth=edgewidth)
            plt.gca().add_patch(poly)

            # Add a text label showing the predicates that hold in this partion

            for keys in inverse_label_dict.keys():
                if keys not in labels:
                    print_label = inverse_label_dict[keys]

            center_point = region.ChebyshevCenter()
            plt.text(center_point[0], center_point[1],
                     ", ".join(['%s'] * len(print_label)) % tuple(print_label),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=12, color='black')

        # Use equal axes so square things look square
        plt.axis('equal')

        # Set the background color
        plt.gca().set_facecolor(background)

if __name__ == '__main__':

    import time
    import numpy as np
    import matplotlib.pyplot as plt

    from pydrake.geometry.optimization import HPolyhedron
    from Time_Automaton import Time_Automaton

    import os

    os.environ["MOSEKLM_LICENSE_FILE"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mosek.lic"
    )

    # ts = Transition_system_of_convex_sets(2)
    #
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([-2, 0], [5, 10]), set(['W', 'b', 'd']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([5, 4], [7, 6]), set(['W', 'd']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([7, 2.1], [9, 10]), set(['W', 'b', 'd']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([7, 0], [9, 2.1]), set(['W', 'b']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([9, 0], [11, 2]), set(['W', 'f', 'b', 'd']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([-2, 0], [0, 2]), set(['W', 'a', 'b', 'd']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([-2, 8], [0, 10]), set(['W', 'e', 'b', 'd']))
    #
    # ts.AddEdgesFromIntersections()
    # feasible_list = ts.feasible_list_and_label_region_construction()
    #
    # TA_1 = Time_Automaton('U', [10, 0.5, 9.5], [set(['W']), set(['W', 'b']), set(['W', 'a'])], 0, feasible_list)  # bU[0,4]a
    # TA_2 = Time_Automaton('U', [10, 0.5, 9.5], [set(['W']), set(['W', 'd']), set(['W', 'e'])], 1, feasible_list)  # dU[5,8]e
    # TA_3 = Time_Automaton('U', [10, 0.5, 9.5], [set(['W']), set(['W']), set(['W', 'f'])], 2, feasible_list)  # F[0,10]f
    # TA_start_time = time.time()
    # TA = (TA_1 & TA_2) & TA_3
    # TA_time = time.time() - TA_start_time
    #
    # # Take the product of the TA and the transition system to produce a graph of
    # # convex sets
    # start_point = [4.0, 9.0]
    # order = 3
    # continuity = 2
    # product_start_time = time.time()
    # bgcs = ts.Product(TA, start_point, order, continuity)
    # product_time = time.time() - product_start_time
    #
    # # Solve the planning problem
    # bgcs.AddVelocityConstraint([-5, -5], [5, 5])
    # bgcs.AddLengthCost(norm="L2")
    # bgcs.AddDerivativeCost(degree=1, weight=1.0,norm="L2")
    # bgcs.AddDerivativeCost(degree=2, weight=1.0,norm="L2")
    # solve_start_time = time.time()
    # res = bgcs.SolveShortestPath(
    #     convex_relaxation=True,
    #     preprocessing=True,
    #     verbose=False,
    #     max_rounded_paths=10,
    #     solver="mosek")
    # solve_time = time.time() - solve_start_time
    #
    # if res.is_success():
    #     # Plot the resulting trajectory
    #     color_dict = {
    #         "white": [[]],
    #         "#2077B4": [["goal"]],
    #         "#F14732": [["door1"], ["door2"]],
    #         "#80BF80": [["key1"], ["key2"]]
    #     }
    #     ts.visualize(color_dict, background='black', alpha=1.0)
    #     bgcs.PlotSolution(res, plot_control_points=False, plot_path=True)
    #
    #     plt.gca().xaxis.set_visible(False)
    #     plt.gca().yaxis.set_visible(False)
    #
    #     # Print timing infos
    #     print("\n")
    #     print("Solve Times:")
    #     print("    STL --> TA    : ", TA_time)
    #     print("    TS x TA = GCS : ", product_time)
    #     print("    GCS solve      : ", solve_time)
    #     print("    Total          : ", TA_time + product_time + solve_time)
    #     print("")
    #
    #     print("GCS vertices: ", len(bgcs.vertices))
    #     print("GCS edges: ", len(bgcs.edges))
    #
    #     # Make an animation of the trajectory
    #     bgcs.AnimateSolution(res, save=False, filename='media/key_door.gif')
    #
    #     plt.show()
    # else:
    #     print("Optimization failed!")

    # # second test interval: 0.001 no bound
    # ts = Transition_system_of_convex_sets(2)
    #
    # # # # ts.add_convex_set(
    # # #    HPolyhedron.MakeBox([0, 0], [10, 10]), set(['W']))  # test
    # #
    # # # ts.add_convex_set(
    # # #    HPolyhedron.MakeBox([0, 0], [10, 10]), set(['W', 'e', 'd']))  # test
    # #
    # # # ts.add_convex_set(
    # # #     HPolyhedron.MakeBox([1., 6.], [4., 9.]), set(['W', 'a', 'e']))  # first GF
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([0., 0.], [1., 9.]), set(['W', 'e', 'd']))  # first GF left
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([0., 9.], [10., 10.]), set(['W', 'e', 'd']))  # first GF up
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([4., 6.], [10., 9.]), set(['W', 'e', 'd']))  # first GF right second GF up
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([1., 4.], [7., 6.]), set(['W', 'e', 'd']))  # first GF down second GF left
    # # # ts.add_convex_set(
    # # #     HPolyhedron.MakeBox([7., 4.], [9., 6.]), set(['W', 'b', 'd']))  # second GF
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([9., 0.], [10., 6.]), set(['W', 'e', 'd']))  # second GF right
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([4., 0.], [9., 4.]), set(['W', 'e', 'd']))  # second GF down
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([1., 0.], [4., 1.]), set(['W', 'e', 'd']))  # FG down
    # # # ts.add_convex_set(
    # # #    HPolyhedron.MakeBox([1., 1.], [4., 4.]), set(['W', 'c', 'e', 'd']))  # third FG
    #
    # ts.add_convex_set(
    #    HPolyhedron.MakeBox([0., 6.], [10., 10.]), set(['W', 'e', 'd']))  # test
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([0., 0.], [10., 4.]), set(['W', 'e', 'd']))  # test
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([0., 0.], [4., 10.]), set(['W', 'e', 'd']))  # test
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([6., 0.], [10., 10.]), set(['W', 'e', 'd']))  # test
    #
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([1., 6.], [4., 9.]), set(['W', 'a', 'e']))  # first GF
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([7., 4.], [9., 6.]), set(['W', 'b', 'd']))  # second GF
    # ts.add_convex_set(
    #    HPolyhedron.MakeBox([1., 1.], [4., 4.]), set(['W', 'c', 'e', 'd']))  # third FG
    #
    # ts.AddEdgesFromIntersections()
    # # print(len(ts.nodes))
    # # print(len(ts.edges))
    # feasible_list = ts.feasible_list_and_label_region_construction()
    #
    # # TA_1 = Time_Automaton('GF', [10, 0.5, 5, 0.1, 3], [set(['W']), set(['W', 'a']), set(['W', 'd'])], 0, feasible_list, 'c1', repeat_number=2)  # bU[0,4]a
    # # TA_2 = Time_Automaton('GF', [10, 1.5, 6, 0.1, 3], [set(['W']), set(['W', 'b']), set(['W', 'e'])], 1, feasible_list, 'c2', repeat_number=1)  # dU[5,8]e
    # # TA_3 = Time_Automaton('FG', [10, 7.5, 7.9, 0.1, 2], [set(['W']), set(['W', 'c'])], 2, feasible_list, 'c3')  # F[0,10]f
    # TA_1 = Time_Automaton('GF', [10, 0., 5., 0., 3.], [set(['W']), set(['W', 'a']), set(['W', 'd'])], 0, feasible_list, 'c1', repeat_number=2)  # bU[0,4]a
    # TA_2 = Time_Automaton('GF', [10, 1., 6., 0., 3.], [set(['W']), set(['W', 'b']), set(['W', 'e'])], 1, feasible_list, 'c2', repeat_number=2)  # dU[5,8]e
    # TA_3 = Time_Automaton('FG', [10, 7., 8., 0., 2.], [set(['W']), set(['W', 'c'])], 2, feasible_list, 'c3')  # F[0,10]f
    # # TA_1 = Time_Automaton('U', [10, 4.5, 6.5], [set(['W']), set(['W']), set(['W', 'c'])], 0, feasible_list)
    # #TA_2 = Time_Automaton('G', [13, 4, 5], [set(['W']), set(['W', 'b'])], 1, feasible_list)
    # #TA_2 = Time_Automaton('U', [10, 4, 5], [set(['W']), set(['W']), set(['W', 'b'])], 1, feasible_list)
    # # TA_3 = Time_Automaton('U', [10, 7.5, 9], [set(['W']), set(['W']), set(['W', 'a'])], 2, feasible_list)
    # # TA_1 = Time_Automaton('U', [10, 0.1, 9], [set(['W']), set(['W']), set(['W', 'c'])], 0, feasible_list)
    # # TA_2 = Time_Automaton('U', [10, 0.1, 9],  [set(['W']), set(['W']), set(['W', 'b'])], 1, feasible_list)
    # # TA_3 = Time_Automaton('U', [10, 0.1, 9], [set(['W']), set(['W']), set(['W', 'a'])], 2, feasible_list)
    # TA_start_time = time.time()
    # TA = (TA_1 & TA_2) & TA_3
    # #TA = (TA_1 & TA_3)
    # #TA = TA_1 & TA_2
    # # print(TA.activate_time_var)
    # #TA = TA_3 #& TA_1
    # #TA.state_time_bound = dict()
    # TA_time = time.time() - TA_start_time
    # # for node in TA.nodes():
    # #     print(node, TA.state_time_bound[node])
    #
    # #print(len(TA.nodes))
    # # Take the product of the TA and the transition system to produce a graph of
    # # convex sets
    # start_point = [4.5, 9.5]
    # order = 3
    # continuity = 2
    # product_start_time = time.time()
    # bgcs = ts.Product(TA, start_point, order, continuity,[-10., -10.], [10., 10.])
    # #bgcs = ts.Product(TA, start_point, order, continuity)
    # product_time = time.time() - product_start_time
    #
    # # Solve the planning problem
    # #bgcs.AddVelocityConstraint([-10., -10.], [10., 10.])
    # bgcs.AddLengthCost(norm="L1")
    # #bgcs.AddDerivativeCost(degree=1, weight=1.0, norm="L2")
    # #bgcs.AddDerivativeCost(degree=2, weight=1.0, norm="L1")
    # solve_start_time = time.time()
    # res = bgcs.SolveShortestPath(
    #     convex_relaxation=True,
    #     preprocessing=False,
    #     verbose=True,
    #     max_rounded_paths=20,
    #     solver="mosek")
    # solve_time = time.time() - solve_start_time
    #
    # # # Plot the resulting trajectory
    # # bgcs.PlotSolution(res, plot_control_points=False, plot_path=True)
    # #
    # # plt.gca().xaxis.set_visible(False)
    # # plt.gca().yaxis.set_visible(False)
    # #
    # # # Print timing infos
    # # print("\n")
    # # print("Solve Times:")
    # # print("    STL --> TA    : ", TA_time)
    # # print("    TS x TA = GCS : ", product_time)
    # # print("    GCS solve      : ", solve_time)
    # # print("    Total          : ", TA_time + product_time + solve_time)
    # # print("")
    # #
    # # print("GCS vertices: ", len(bgcs.vertices))
    # # print("GCS edges: ", len(bgcs.edges))
    #
    # # Make an animation of the trajectory
    # # bgcs.AnimateSolution(res, save=False, filename='media/key_door.gif')
    #
    # plt.show()
    #
    # if res.is_success():
    #     # Plot the resulting trajectory
    #     color_dict = {
    #         "white": [[]],
    #         "#2077B4": [set(['W', 'c', 'e', 'd']), set(['W', 'a', 'e']), set(['W', 'b', 'd'])],
    #         "#F14732": [["door1"], ["door2"]],
    #         "#80BF80": [["key1"], ["key2"]]
    #     }
    #     ts.visualize(color_dict, background='black', alpha=1.0)
    #     bgcs.PlotSolution(res, plot_control_points=False, plot_path=True)
    #
    #     plt.gca().xaxis.set_visible(False)
    #     plt.gca().yaxis.set_visible(False)
    #
    #     # Print timing infos
    #     print("\n")
    #     print("Solve Times:")
    #     print("    STL --> TA    : ", TA_time)
    #     print("    TS x TA = GCS : ", product_time)
    #     print("    GCS solve      : ", solve_time)
    #     print("    Total          : ", TA_time + product_time + solve_time)
    #     print("")
    #
    #     print("GCS vertices: ", len(bgcs.vertices))
    #     print("GCS edges: ", len(bgcs.edges))
    #
    #     # Make an animation of the trajectory
    #     bgcs.AnimateSolution(res, save=False, filename='media/key_door.gif')
    #
    #     plt.show()
    # else:
    #     print("Optimization failed!")

    # third test # 1e-6 no b
    ts = Transition_system_of_convex_sets(2)

    ts.add_convex_set(
        HPolyhedron.MakeBox([0, 0], [2, 5]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([0, 8], [10, 10]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([0, 5.5], [4.5, 7.5]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([8, 0], [10, 2]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([8, 2.5], [10, 7.5]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([2.5, 0], [7.5, 5]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([5, 5], [7.5, 7.5]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5']))

    ts.add_convex_set(
        HPolyhedron.MakeBox([2, 0.5], [2.5, 1.5]), set(['W', 'a2', 'a3', 'a4', 'a5']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([0.5, 5], [1.5, 5.5]), set(['W', 'a1', 'a2', 'a3', 'a5']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([7.5, 6], [8, 7]), set(['W', 'a1', 'a3', 'a4', 'a5']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([8.5, 2], [9.5, 2.5]), set(['W', 'a1', 'a2', 'a4', 'a5']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([8.5, 7.5], [9.5, 8]), set(['W', 'a1', 'a2', 'a3', 'a4']))

    ts.add_convex_set(
        HPolyhedron.MakeBox([0.5, 0.5], [1.5, 1.5]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5', 'g3']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([6, 0.5], [7, 1.5]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5', 'g1']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([8.5, 0.5], [9.5, 1.5]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5', 'g4']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([3, 3.5], [4, 4.5]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5', 'g2']))
    ts.add_convex_set(
        HPolyhedron.MakeBox([3, 6], [4, 7]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5', 'g5']))

    ts.add_convex_set(
        HPolyhedron.MakeBox([0.5, 8.5], [1.5, 9.6]), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5', 'g']))

    ts.AddEdgesFromIntersections()
    feasible_list = ts.feasible_list_and_label_region_construction()
    # Convert the specification to a TA
    print("Converting to TA")
    TA_start_time = time.time()
    # TA_1 = Time_Automaton('U', [10, 0.5, 1], [set(['W']), set(['W', 'a1']), set(['W', 'g1'])], 0,
    #                       feasible_list)  # bU[0,4]a
    # TA_2 = Time_Automaton('U', [10, 1.5, 2.5], [set(['W']), set(['W', 'a2']), set(['W', 'g2'])], 1,
    #                       feasible_list)  # dU[5,8]e
    # TA_3 = Time_Automaton('U', [10, 3.5, 5.5], [set(['W']), set(['W', 'a3']), set(['W', 'g3'])], 2,
    #                       feasible_list)  # F[0,10]f
    # TA_4 = Time_Automaton('U', [10, 6.5, 9.5], [set(['W']), set(['W', 'a4']), set(['W', 'g4'])], 3,
    #                       feasible_list)  # dU[5,8]e
    # TA_5 = Time_Automaton('U', [10, 6.5, 9.5], [set(['W']), set(['W', 'a5']), set(['W', 'g5'])], 4,
    #                       feasible_list)  # F[0,10]f
    # TA_6 = Time_Automaton('U', [10, 0.5, 9.5], [set(['W']), set(['W']), set(['W', 'g'])], 5,
    #                       feasible_list)  # F[0,10]f
    TA_1 = Time_Automaton('U', [10, 0.5, 9.5], [set(['W']), set(['W', 'a1']), set(['W', 'g1'])], 0,
                          feasible_list)  # bU[0,4]a
    TA_2 = Time_Automaton('U', [10, 0.5, 9.5], [set(['W']), set(['W', 'a2']), set(['W', 'g2'])], 1,
                          feasible_list)  # dU[5,8]e
    TA_3 = Time_Automaton('U', [10, 0.5, 9.5], [set(['W']), set(['W', 'a3']), set(['W', 'g3'])], 2,
                          feasible_list)  # F[0,10]f
    TA_4 = Time_Automaton('U', [10, 0.5, 9.5], [set(['W']), set(['W', 'a4']), set(['W', 'g4'])], 3,
                          feasible_list)  # dU[5,8]e
    TA_5 = Time_Automaton('U', [10, 0.5, 9.5], [set(['W']), set(['W', 'a5']), set(['W', 'g5'])], 4,
                          feasible_list)  # F[0,10]f
    TA_6 = Time_Automaton('U', [10, 0.5, 9.5], [set(['W']), set(['W']), set(['W', 'g'])], 5,
                          feasible_list)  # F[0,10]f
    TA = (((((TA_1 & TA_2) & TA_3) & TA_4) & TA_5) & TA_6)
    #TA.time_bound_for_state()
    TA_time = time.time() - TA_start_time

    # Take the product of the DFA and the transition system to produce a graph of
    # convex sets
    print("Computing product GCS")
    start_point = [5.0, 2.5]
    order = 4
    continuity = 2
    product_start_time = time.time()
    bgcs = ts.Product(TA, start_point, order, continuity,[-10, -10], [10, 10])
    product_time = time.time() - product_start_time

    # Solve the planning problem
    print("Solving Shortest Path")
    bgcs.AddLengthCost(norm="L1")
    bgcs.AddDerivativeCost(degree=1, weight=1.0, norm="L1")
    bgcs.AddDerivativeCost(degree=2, weight=1.0, norm="L1")
    #bgcs.AddVelocityConstraint([-40, -40], [40, 40])
    solve_start_time = time.time()
    res = bgcs.SolveShortestPath(
        convex_relaxation=True,
        preprocessing=False,
        verbose=False,
        max_rounded_paths=10,
        solver="mosek")
    solve_time = time.time() - solve_start_time

    if res.is_success():
        # Plot the resulting trajectory
        color_dict = {
            "white": [[]],
            "#2077B4": [["goal"]],
            "#F14732": [["d1"], ["d2"], ["d3"], ["d4"], ["d5"]],
            "#80BF80": [["k1"], ["k2"], ["k3"], ["k4"], ["k5"]]
        }
        # ts.visualize(color_dict, background='black', edgewidth=0.0, alpha=1.0)
        bgcs.PlotSolution(res, plot_control_points=False, plot_path=True)
        plt.xlim((-0.7, 10.7))
        plt.ylim((-0.7, 10.7))
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)

        # Print timing infos
        print("\n")
        print("Solve Times:")
        print("    STL --> TA    : ", TA_time)
        print("    TS x TA = GCS : ", product_time)
        print("    GCS solve      : ", solve_time)
        print("    Total          : ", TA_time + product_time + solve_time)
        print("")

        print("GCS vertices: ", len(bgcs.vertices))
        print("GCS edges: ", len(bgcs.edges))

        # Make an animation of the trajectory
        bgcs.AnimateSolution(res, save=False, filename='media/door_puzzle.gif')

        plt.show()
    else:
        print("Optimization failed!")

    # # forth test
    # ts = Transition_system_of_convex_sets(2)
    #
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([0, 0], [4, 10]), set(['W']))
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([0, 0], [10, 4]), set(['W']))
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([6, 0], [10, 10]), set(['W']))
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([0, 6], [10, 10]), set(['W']))
    #
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([0., 0.], [4., 4.]), set(['W']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([0., 4.], [4., 6.]), set(['W']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([0., 6.], [4., 10.]), set(['W']))
    # # ts.add_convex_set(
    # #     HPolyhedron.MakeBox([0, 0], [4, 10]), set(['W']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([4., 0.], [6., 4.]), set(['W']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([6., 0.], [10., 4.]), set(['W']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([6., 4.], [10., 6.]), set(['W']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([6., 6.], [10., 10.]), set(['W']))
    # # ts.add_convex_set(
    # #         HPolyhedron.MakeBox([6, 0], [10, 10]), set(['W']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([4., 6.], [6., 10.]), set(['W']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([1., 1.], [3., 3.]), set(['W', 'a']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([8., 1.], [9., 2.]), set(['W', 'b']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([1., 8.], [2., 9.]), set(['W', 'c']))
    # ts.add_convex_set(
    #     HPolyhedron.MakeBox([8., 8.], [9., 9.]), set(['W', 'd']))
    #
    # ts.AddEdgesFromIntersections()
    # feasible_list = ts.feasible_list_and_label_region_construction()
    #
    # TA_1 = Time_Automaton('U', [10, 0.1, 9.5], [set(['W']), set(['W']), set(['W', 'a'])], 0, feasible_list)  # bU[0,4]a
    # TA_2 = Time_Automaton('U', [10, 0.1, 9.5], [set(['W']), set(['W']), set(['W', 'b'])], 1, feasible_list)  # dU[5,8]e
    # TA_3 = Time_Automaton('U', [10, 0.1, 9.5], [set(['W']), set(['W']), set(['W', 'c'])], 2, feasible_list)  # F[0,10]f
    # TA_4 = Time_Automaton('U', [10, 0.1, 9.5], [set(['W']), set(['W']), set(['W', 'd'])], 3, feasible_list)  # F[0,10]f
    # #
    # TA_start_time = time.time()
    # TA = ((TA_1 & TA_2) & TA_3) & TA_4
    # # TA = Time_Automaton('seq', [30, 29.9],
    # #                     [set(['W']), set(['W', 'a']), set(['W', 'b']), set(['W', 'd']), set(['W', 'c'])], 4,
    # #                     feasible_list)
    # TA_time = time.time() - TA_start_time
    #
    # # Take the product of the TA and the transition system to produce a graph of
    # # convex sets
    # start_point = [4.5, 9.0]
    # order = 3
    # continuity = 2
    # product_start_time = time.time()
    # bgcs = ts.Product(TA, start_point, order, continuity)
    # product_time = time.time() - product_start_time
    #
    # # Solve the planning problem
    # bgcs.AddVelocityConstraint([-20, -20], [20, 20])
    # bgcs.AddLengthCost(norm="L1")
    # bgcs.AddDerivativeCost(degree=1, weight=1.0,norm="L1")
    # bgcs.AddDerivativeCost(degree=2, weight=1.0,norm="L1")
    # solve_start_time = time.time()
    # res = bgcs.SolveShortestPath(
    #     convex_relaxation=True,
    #     preprocessing=False,
    #     verbose=True,
    #     max_rounded_paths=10,
    #     solver="mosek")
    # solve_time = time.time() - solve_start_time
    #
    # if res.is_success():
    #     # Plot the resulting trajectory
    #     color_dict = {
    #         "white": [[]],
    #         "#2077B4": [["goal"]],
    #         "#F14732": [["door1"], ["door2"]],
    #         "#80BF80": [["key1"], ["key2"]]
    #     }
    #     ts.visualize(color_dict, background='black', alpha=1.0)
    #     bgcs.PlotSolution(res, plot_control_points=True, plot_path=True)
    #
    #     plt.gca().xaxis.set_visible(False)
    #     plt.gca().yaxis.set_visible(False)
    #
    #     # Print timing infos
    #     print("\n")
    #     print("Solve Times:")
    #     print("    STL --> TA    : ", TA_time)
    #     print("    TS x TA = GCS : ", product_time)
    #     print("    GCS solve      : ", solve_time)
    #     print("    Total          : ", TA_time + product_time + solve_time)
    #     print("")
    #
    #     print("GCS vertices: ", len(bgcs.vertices))
    #     print("GCS edges: ", len(bgcs.edges))
    #
    #     # Make an animation of the trajectory
    #     bgcs.AnimateSolution(res, save=False, filename='media/key_door.gif')
    #
    #     plt.show()
    # else:
    #     print("Optimization failed!")
