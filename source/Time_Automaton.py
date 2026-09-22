import time
import networkx as nx
import numpy as np

class Time_Automaton(nx.DiGraph):
    """
    class of time automaton
    """

    def _set_state_metadata(self, node, region_label, time_bound, active_time_var):
        self.vertices_to_regions_label[node] = set(region_label)
        self.state_time_bound[node] = dict(time_bound)
        self.activate_time_var[node] = set(active_time_var)

    def _set_edge_metadata(self, edge, time_constraint, reset_variables):
        self.edges_time_constraint[edge] = dict(time_constraint)
        self.edges_reset_variables[edge] = set(reset_variables)

    def _build_g_template(self, time_info, location_info, template_index):
        node_list = [0 + 5 * template_index, 1 + 5 * template_index, 2 + 5 * template_index]
        self.add_nodes_from(node_list)
        self.add_edges_from([(node_list[0], node_list[1]), (node_list[1], node_list[2])])
        if time_info[1] == 0:
            self.initial_vertices = [node_list[1]]
        else:
            self.initial_vertices = [node_list[0]]
        self.time_variables.add('c0')

        self._set_edge_metadata((node_list[0], node_list[1]), {'c0': (0, time_info[1])}, set())
        self._set_edge_metadata((node_list[1], node_list[2]), {'c0': (time_info[2], time_info[0])}, set())

        self._set_state_metadata(node_list[0], location_info[0], {'c0': (0, time_info[1])}, {'c0'})
        self._set_state_metadata(node_list[1], location_info[1],
                                 {'c0': (time_info[1] - 1e-4, time_info[2] + 1e-4)}, {'c0'})
        self._set_state_metadata(node_list[2], location_info[0], {'c0': (time_info[2], time_info[0])}, {'c0'})

        self.accepting_vertices = [node_list[2]]

    def _build_u_template(self, time_info, location_info, template_index):
        node_list = [0 + 5 * template_index, 1 + 5 * template_index, 2 + 5 * template_index]
        self.add_nodes_from(node_list)
        self.add_edges_from([(node_list[0], node_list[1]), (node_list[1], node_list[2])])
        self.initial_vertices = [node_list[0]]
        self.time_variables.add('c0')

        self._set_edge_metadata((node_list[0], node_list[1]), {'c0': (time_info[1], time_info[2])}, set())
        self._set_edge_metadata((node_list[1], node_list[2]), {'c0': (0, time_info[0])}, set())

        self._set_state_metadata(node_list[0], location_info[1], {'c0': (0, time_info[2])}, {'c0'})
        self._set_state_metadata(node_list[1], location_info[2],
                                 {'c0': (time_info[1] - 1e-4, time_info[2] + 1e-4)}, {'c0'})
        self._set_state_metadata(node_list[2], location_info[0], {'c0': (time_info[1], time_info[0])}, {'c0'})

        self.accepting_vertices = [node_list[1], node_list[2]]

    def _build_fg_template(self, time_info, location_info, template_index, time_variable_name):
        node_list = [0 + 5 * template_index, 1 + 5 * template_index, 2 + 5 * template_index]
        self.add_nodes_from(node_list)
        self.add_edges_from([(node_list[0], node_list[1]), (node_list[1], node_list[2])])
        self.initial_vertices = [node_list[0]]
        self.time_variables.add('c0')
        self.time_variables.add(time_variable_name)

        self._set_edge_metadata(
            (node_list[0], node_list[1]),
            {'c0': (time_info[1] + time_info[3], time_info[2] + time_info[3]),
             time_variable_name: (0, time_info[0])},
            {time_variable_name},
        )
        self._set_edge_metadata(
            (node_list[1], node_list[2]),
            {'c0': (0, time_info[0]),
             time_variable_name: (time_info[4] - time_info[3], time_info[0])},
            set(),
        )

        self._set_state_metadata(
            node_list[0],
            location_info[0],
            {'c0': (0, time_info[2] + time_info[3]), time_variable_name: (0, time_info[2] + time_info[3])},
            {'c0'},
        )
        self._set_state_metadata(
            node_list[1],
            location_info[1],
            {'c0': (time_info[1] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
            {'c0', time_variable_name},
        )
        self._set_state_metadata(
            node_list[2],
            location_info[0],
            {'c0': (time_info[1] + time_info[4], time_info[0]),
             time_variable_name: (time_info[4] - time_info[3], time_info[0])},
            {'c0'},
        )

        self.accepting_vertices = [node_list[2]]

    def _build_gf_template(self, time_info, location_info, template_index, time_variable_name, repeat_number):
        node_list = [0 + 5 * template_index, 1 + 5 * template_index, 2 + 5 * template_index, 4 + 5 * template_index]
        self.add_nodes_from(node_list)
        edge_0 = (node_list[0], node_list[3])
        edge_1 = (node_list[3], node_list[1])
        edge_2 = (node_list[1], node_list[2])
        self.add_edges_from([edge_0, edge_1, edge_2])
        if time_info[1] + time_info[3] == 0:
            self.initial_vertices = [node_list[3]]
        else:
            self.initial_vertices = [node_list[0]]
        self.time_variables.add('c0')
        self.time_variables.add(time_variable_name)

        self._set_edge_metadata(
            edge_0,
            {'c0': (0, time_info[1] + time_info[3]), time_variable_name: (0, time_info[0])},
            {time_variable_name},
        )
        self._set_edge_metadata(
            edge_1,
            {'c0': (0, time_info[0]), time_variable_name: (0, time_info[4] - time_info[3])},
            set(),
        )
        self._set_edge_metadata(
            edge_2,
            {'c0': (time_info[2] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
            set(),
        )

        self._set_state_metadata(
            node_list[0],
            location_info[0],
            {'c0': (0, time_info[1] + time_info[3]), time_variable_name: (0, time_info[1] + time_info[3])},
            {'c0'},
        )
        self._set_state_metadata(
            node_list[1],
            location_info[1],
            {'c0': (time_info[1] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
            {'c0'},
        )
        self._set_state_metadata(
            node_list[2],
            location_info[0],
            {'c0': (time_info[2] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
            {'c0'},
        )
        self._set_state_metadata(
            node_list[3],
            location_info[0],
            {'c0': (0, time_info[1] + time_info[4]), time_variable_name: (0, time_info[4] - time_info[3])},
            {'c0', time_variable_name},
        )

        self.accepting_vertices = [node_list[2]]

        # The generic TA product avoids revisiting the same discrete
        # product state in a single exploration path to keep the search
        # space finite. Therefore, loop-like STL semantics are handled by
        # explicitly unrolling additional copies of the relevant states.
        for num in range(repeat_number - 1):
            node_R1 = (str(template_index), 1 + 5 * template_index, num)
            node_R2 = (str(template_index), 3 + 5 * template_index, num)
            self.add_node(node_R1)
            self.add_node(node_R2)
            if num == 0:
                past_node = node_list[1]
            else:
                past_node = (str(template_index), 1 + 5 * template_index, num - 1)
            add_edge_list = [(past_node, node_R2), (node_R2, node_R1), (node_R1, node_list[2])]
            self.add_edges_from(add_edge_list)
            self._set_edge_metadata(
                add_edge_list[0],
                {'c0': (0, time_info[0]), time_variable_name: (0, time_info[0])},
                {time_variable_name},
            )
            self._set_edge_metadata(
                add_edge_list[1],
                {'c0': (0, time_info[0]), time_variable_name: (0, time_info[4] - time_info[3])},
                set(),
            )
            self._set_edge_metadata(
                add_edge_list[2],
                {'c0': (time_info[2] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
                set(),
            )

            self._set_state_metadata(
                node_R1,
                location_info[1],
                {'c0': (time_info[1] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
                {'c0'},
            )
            self._set_state_metadata(
                node_R2,
                location_info[2],
                {'c0': (time_info[1] + time_info[3], time_info[0]), time_variable_name: (0, time_info[4] - time_info[3])},
                {'c0', time_variable_name},
            )

    def _build_gfl_template(self, time_info, location_info, template_index, time_variable_name, repeat_number):
        node_list = [0 + 5 * template_index, 1 + 5 * template_index, 2 + 5 * template_index, 4 + 5 * template_index, 3 + 5 * template_index]
        self.add_nodes_from(node_list)
        edge_0 = (node_list[0], node_list[3])
        edge_1 = (node_list[3], node_list[1])
        edge_2 = (node_list[1], node_list[2])
        edge_3 = (node_list[1], node_list[4])
        edge_4 = (node_list[4], node_list[1])
        self.add_edges_from([edge_0, edge_1, edge_2, edge_3, edge_4])
        if time_info[1] + time_info[3] == 0:
            self.initial_vertices = [node_list[3]]
        else:
            self.initial_vertices = [node_list[0]]
        self.time_variables.add('c0')
        self.time_variables.add(time_variable_name)

        self._set_edge_metadata(
            edge_0,
            {'c0': (0, time_info[1] + time_info[3]), time_variable_name: (0, time_info[0])},
            {time_variable_name},
        )
        self._set_edge_metadata(
            edge_1,
            {'c0': (0, time_info[0]), time_variable_name: (0, time_info[4] - time_info[3])},
            set(),
        )
        self._set_edge_metadata(
            edge_2,
            {'c0': (time_info[2] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
            set(),
        )
        self._set_edge_metadata(
            edge_3,
            {'c0': (0, time_info[0]), time_variable_name: (0, time_info[0])},
            {time_variable_name},
        )
        self._set_edge_metadata(
            edge_4,
            {'c0': (0, time_info[0]), time_variable_name: (0, time_info[4] - time_info[3])},
            set(),
        )

        self._set_state_metadata(
            node_list[0],
            location_info[0],
            {'c0': (0, time_info[1] + time_info[3]), time_variable_name: (0, time_info[1] + time_info[3])},
            {'c0'},
        )
        self._set_state_metadata(
            node_list[1],
            location_info[1],
            {'c0': (time_info[1] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
            {'c0'},
        )
        self._set_state_metadata(
            node_list[2],
            location_info[0],
            {'c0': (time_info[2] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
            {'c0'},
        )
        self._set_state_metadata(
            node_list[3],
            location_info[0],
            {'c0': (0, time_info[1] + time_info[4]), time_variable_name: (0, time_info[4] - time_info[3])},
            {'c0', time_variable_name},
        )
        self._set_state_metadata(
            node_list[4],
            location_info[2],
            {'c0': (time_info[1] + time_info[3], time_info[0]), time_variable_name: (0, time_info[4] - time_info[3])},
            {'c0', time_variable_name},
        )

        self.accepting_vertices = [node_list[2]]

        # See the note in the GF template above: repeated visits are
        # represented by bounded template unfolding instead of generic
        # product-state revisits.
        for num in range(repeat_number - 1):
            node_R1 = (str(template_index), 1 + 5 * template_index, num)
            node_R2 = (str(template_index), 3 + 5 * template_index, num)
            self.add_node(node_R1)
            self.add_node(node_R2)
            if num == 0:
                past_node = node_list[1]
            else:
                past_node = (str(template_index), 1 + 5 * template_index, num - 1)
            add_edge_list = [(past_node, node_R2), (node_R2, node_R1), (node_R1, node_list[2]), (node_R1, node_R2)]
            self.add_edges_from(add_edge_list)
            self._set_edge_metadata(
                add_edge_list[0],
                {'c0': (0, time_info[0]), time_variable_name: (0, time_info[0])},
                {time_variable_name},
            )
            self._set_edge_metadata(
                add_edge_list[1],
                {'c0': (0, time_info[0]), time_variable_name: (0, time_info[4] - time_info[3])},
                set(),
            )
            self._set_edge_metadata(
                add_edge_list[2],
                {'c0': (time_info[2] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
                set(),
            )
            self._set_edge_metadata(
                add_edge_list[3],
                {'c0': (0, time_info[0]), time_variable_name: (0, time_info[0])},
                {time_variable_name},
            )

            self._set_state_metadata(
                node_R1,
                location_info[1],
                {'c0': (time_info[1] + time_info[3], time_info[0]), time_variable_name: (0, time_info[0])},
                {'c0'},
            )
            self._set_state_metadata(
                node_R2,
                location_info[2],
                {'c0': (time_info[1] + time_info[3], time_info[0]), time_variable_name: (0, time_info[4] - time_info[3])},
                {'c0', time_variable_name},
            )

    def _build_res_template(self, time_info, location_info, template_index, time_variable_name):
        self.time_variables.add('c0')
        self.time_variables.add(time_variable_name)
        self.initial_vertices = [(str(template_index), 0, 0)]

        for num in range(2):
            node_list = [(str(template_index), i, num) for i in range(5)]
            self.add_nodes_from(node_list)
            edge_0 = (node_list[0], node_list[1])
            edge_1 = (node_list[1], node_list[2])
            edge_2 = (node_list[2], node_list[3])
            edge_3 = (node_list[3], node_list[4])
            self.add_edges_from([edge_0, edge_1, edge_2, edge_3])

            self._set_edge_metadata(
                edge_0,
                {'c0': (0, time_info[1]), time_variable_name: (0, time_info[1])},
                {time_variable_name},
            )
            self._set_edge_metadata(
                edge_1,
                {'c0': (0, time_info[1]), time_variable_name: (0, time_info[2])},
                set(),
            )
            self._set_edge_metadata(
                edge_2,
                {'c0': (0, time_info[1]), time_variable_name: (0, time_info[2])},
                set(),
            )
            self._set_edge_metadata(
                edge_3,
                {'c0': (0, time_info[0]), time_variable_name: (0, time_info[2])},
                set(),
            )

            self._set_state_metadata(
                node_list[0],
                location_info[3],
                {'c0': (0, time_info[1]), time_variable_name: (0, time_info[0])},
                {'c0'},
            )
            self._set_state_metadata(
                node_list[1],
                location_info[2],
                {'c0': (0, time_info[1]), time_variable_name: (0, time_info[2])},
                {'c0', time_variable_name},
            )
            self._set_state_metadata(
                node_list[2],
                location_info[1][num],
                {'c0': (0, time_info[1]), time_variable_name: (0, time_info[2])},
                {'c0', time_variable_name},
            )
            self._set_state_metadata(
                node_list[3],
                location_info[0],
                {'c0': (0, time_info[1]), time_variable_name: (0, time_info[2])},
                {'c0', time_variable_name},
            )
            self._set_state_metadata(
                node_list[4],
                location_info[4],
                {'c0': (0, time_info[0]), time_variable_name: (0, time_info[0])},
                {'c0'},
            )

            if num != 0:
                past_node = (str(template_index), 4, num - 1)
                new_edge = (past_node, node_list[0])
                self.add_edge(past_node, node_list[0])
                self._set_edge_metadata(
                    new_edge,
                    {'c0': (0, time_info[1]), time_variable_name: (0, time_info[1])},
                    set(),
                )

        acc_state = (str(template_index), 5, 0)
        self.add_node(acc_state)
        self._set_state_metadata(
            acc_state,
            location_info[0],
            {'c0': (time_info[1], time_info[0]), time_variable_name: (0, time_info[0])},
            {'c0'},
        )
        self.accepting_vertices = [acc_state]
        new_edge = (node_list[-1], acc_state)
        self.add_edge(node_list[-1], acc_state)
        self._set_edge_metadata(
            new_edge,
            {'c0': (time_info[1], time_info[0]), time_variable_name: (0, time_info[0])},
            set(),
        )

    def _build_seq_template(self, time_info, location_info, template_index):
        self.time_variables.add('c0')
        self.initial_vertices = [(str(template_index), 0, 0)]

        for num in range(4):
            node_list = [(str(template_index), i, num) for i in range(2)]
            self.add_nodes_from(node_list)
            edge_0 = (node_list[0], node_list[1])
            self.add_edges_from([edge_0])

            self._set_edge_metadata(
                edge_0,
                {'c0': (0, time_info[1])},
                set(),
            )

            self._set_state_metadata(
                node_list[0],
                location_info[0],
                {'c0': (0, time_info[1])},
                {'c0'},
            )
            self._set_state_metadata(
                node_list[1],
                location_info[1 + num],
                {'c0': (0, time_info[1])},
                {'c0'},
            )

            if num != 0:
                past_node = (str(template_index), 1, num - 1)
                new_edge = (past_node, node_list[0])
                self.add_edge(past_node, node_list[0])
                self._set_edge_metadata(
                    new_edge,
                    {'c0': (0, time_info[1])},
                    set(),
                )

        acc_state = (str(template_index), 2, 0)
        self.add_node(acc_state)
        self._set_state_metadata(
            acc_state,
            location_info[0],
            {'c0': (0, time_info[0])},
            {'c0'},
        )
        self.accepting_vertices = [acc_state]
        new_edge = (node_list[-1], acc_state)
        self.add_edge(node_list[-1], acc_state)
        self._set_edge_metadata(
            new_edge,
            {'c0': (0, time_info[0])},
            set(),
        )

    def __init__(self, operator_type='', time_info=[0], location_info=[], template_index=0, feasible_label=[], time_variable_name='', repeat_number=1):

        """
        Create the time automaton for each template STL.

        Args:
            operator_type: the type of temporal operator, can be F(finally), G(globally), U(until), FG(eventually_always), and GF
            time_info: the time parameter for temporal operator
            location_info: a list of set that contains corresponding labels
            time_variable_name: the name of the addition time variable(used at case FG and GF)
            template_index: int to record the order of template, ensure the uniqueness of TA state
            feasible_label: a list that contains all feasible label(used in product operation)
            repeat_number: bounded unfolding depth for templates that require
                           loop-like behavior (e.g. GF). The generic TA product
                           below only explores acyclic state paths, so repeated
                           visits must be encoded explicitly by the template.
        """

        ## nodes [i,j,k...] and edges [(i,j), (i,k)...] already in class digraph
        super().__init__()
        self.initial_vertices = []  # list of TA initial nodes
        self.time_variables = set()  # active clocks for this TA; 'c0' stores physical time progress
        self.edges_time_constraint = dict()  # edge -> {clock_name: feasible interval at the transition}
        self.edges_reset_variables = dict()  # edge -> clocks reset to zero when taking the transition
        self.vertices_to_regions_label = dict()  # node -> region label set that must hold while staying in the node
        self.accepting_vertices = []  # list of TA accepting nodes
        self.feasible_label = feasible_label
        self.end_time = time_info[0]
        self.state_time_bound = dict()  # node -> admissible interval for each clock while staying in the node
        self.activate_time_var = dict()  # node -> clocks that should propagate through outgoing TS transitions
        # Reporting metadata survives TA products/unions.  ``states_after`` is
        # the template size after bounded GF/GFL unfolding, before composition
        # pruning; len(self.nodes) remains the authoritative final |S|.
        self.template_metadata = []

        builder_map = {
            'G': lambda: self._build_g_template(time_info, location_info, template_index),
            'U': lambda: self._build_u_template(time_info, location_info, template_index),
            'FG': lambda: self._build_fg_template(time_info, location_info, template_index, time_variable_name),
            'GF': lambda: self._build_gf_template(time_info, location_info, template_index, time_variable_name, repeat_number),
            'GFL': lambda: self._build_gfl_template(time_info, location_info, template_index, time_variable_name, repeat_number),
            'res': lambda: self._build_res_template(time_info, location_info, template_index, time_variable_name),
            'seq': lambda: self._build_seq_template(time_info, location_info, template_index),
        }

        if operator_type == '':  # the empty string represents the empty TA
            return

        if operator_type not in builder_map:
            raise Exception('No such template to construct TA')

        builder_map[operator_type]()
        unfolded_states = len(self.nodes)
        base_states = unfolded_states
        if operator_type in ('GF', 'GFL'):
            base_states -= 2 * max(0, repeat_number - 1)
        self.template_metadata.append({
            'operator': operator_type,
            'template_index': template_index,
            'repeat_number': repeat_number if operator_type in ('GF', 'GFL') else 1,
            'states_before_unfolding': base_states,
            'states_after_unfolding': unfolded_states,
        })

    def reporting_summary(self):
        """Return the automaton-scale fields used in experiment reports."""
        return {
            'ta_states': len(self.nodes),
            'ta_edges': len(self.edges),
            'clocks': len(self.time_variables),
            'clock_names': sorted(self.time_variables),
            'finite_unfolding': [dict(item) for item in self.template_metadata
                                 if item.get('repeat_number', 1) > 1],
            'templates': [dict(item) for item in self.template_metadata],
        }

    def Prune(self):  # remove the node that can not reach accepting state

        reachable_node = [node for node in self.accepting_vertices]
        new_node = [node for node in self.accepting_vertices]

        while len(new_node) != 0:
            unfold_node = new_node
            new_node = []
            for node in unfold_node:
                for pre_node in self.predecessors(node):
                    if pre_node not in reachable_node:
                        reachable_node.append(pre_node)
                        new_node.append(pre_node)

        for node in set(self.nodes) - set(reachable_node):
            self.remove_node(node)

    def invariance_time_bound(self, TA, time_bound, state_bound):
        delta = self.end_time
        for time_var in TA.time_variables:
            if time_var != 'c0':
                now_del = state_bound[time_var][1] - state_bound[time_var][0]
                if now_del < delta:
                    delta = now_del

        return time_bound+delta

    def check_edge_feasible(self, now_time_cons, TA, edge, state_bound, now_bound):  # use in product, check the edge feasibility

        next_time_cons = dict()
        min_progress = 0
        max_progress = self.end_time
        feasible_flag = True

        for time_var in self.time_variables: #for time_var in TA.product_variable:
            if time_var in TA.time_variables:
                left_bound, right_bound = TA.edges_time_constraint[edge][time_var]
                if min_progress < (left_bound - now_time_cons[time_var][1]):
                    min_progress = (left_bound - now_time_cons[time_var][1])
                if max_progress > (right_bound - now_time_cons[time_var][0]):
                    max_progress = (right_bound - now_time_cons[time_var][0])

        if max_progress <= 0:  # negative means that the edge is infeasible
            feasible_flag = False

        for time_var in self.time_variables: #for time_var in self.self.product_variable:
            if time_var == 'c0':
                next_time_cons[time_var] = (
                max(TA.edges_time_constraint[edge][time_var][0], now_time_cons[time_var][0] + min_progress),
                min(now_time_cons[time_var][1] + max_progress, TA.edges_time_constraint[edge][time_var][1], now_bound))
            elif time_var in TA.time_variables: #if time_var in TA.product_variable:
                next_time_cons[time_var] = (max(TA.edges_time_constraint[edge][time_var][0], now_time_cons[time_var][0] + min_progress),
                     min(now_time_cons[time_var][1] + max_progress, TA.edges_time_constraint[edge][time_var][1]))
            else:
                next_time_cons[time_var] = (
                max(0, now_time_cons[time_var][0] + min_progress), min(self.end_time, now_time_cons[time_var][1] + max_progress))

        for time_var in TA.edges_reset_variables[edge]:
            next_time_cons[time_var] = (0, 0)

        for time_var in self.time_variables:
            cons_bound = next_time_cons[time_var]
            current_state_bound = state_bound[time_var]
            if cons_bound[1] < current_state_bound[0] or current_state_bound[1] < cons_bound[0]:
                feasible_flag = False
            next_time_cons[time_var] = (max(cons_bound[0], current_state_bound[0]), min(cons_bound[1], current_state_bound[1]))

        return next_time_cons, feasible_flag

    def check_label_feasible(self, label):

        for possible_label in self.feasible_label:
            if label <= possible_label:
                return True

        return False

    def add_edge_in_TA(self, TA, now_node, next_node, edge):

        self.edges_reset_variables[(now_node, next_node)] = set(TA.edges_reset_variables[edge])
        next_time_variables_cons = self.augment_time_constraint(self.time_variables, TA.time_variables,
                                                                  TA.edges_time_constraint[edge])
        self.add_edges_from([(now_node, next_node)])
        self.edges_time_constraint[(now_node, next_node)] = next_time_variables_cons

    def augment_time_constraint(self, all_time_variables, partial_time_variables, partial_time_cons):

        augmented_time_cons = partial_time_cons.copy()
        for time_var in all_time_variables - partial_time_variables:
            augmented_time_cons[time_var] = (0, self.end_time)

        return augmented_time_cons

    def check_two_state_feasible(self, left_state, right_state, left_TA, right_TA):

        new_label = left_TA.vertices_to_regions_label[left_state].union(right_TA.vertices_to_regions_label[right_state])
        lower_bound = max(left_TA.state_time_bound[left_state]['c0'][0], right_TA.state_time_bound[right_state]['c0'][0])
        upper_bound = min(left_TA.state_time_bound[left_state]['c0'][1], right_TA.state_time_bound[right_state]['c0'][1])
        if lower_bound < upper_bound:
            label_flag = self.check_label_feasible(new_label)
            if label_flag:
                return True, new_label, (lower_bound, upper_bound)

        return False, None, None

    def recompute_state_time_bound(self, state):

        upper_bound = self.state_time_bound[state]['c0'][1]
        for time_var in self.time_variables:
            current_bound = self.state_time_bound[state][time_var]
            self.state_time_bound[state][time_var] = (current_bound[0], min(current_bound[1], upper_bound))

    def product_from_two_lists(self, left_list, right_list, left_TA, right_TA):

        for left_state in left_list:
            for right_state in right_list:
                if (left_state, right_state) not in self.nodes:
                    feasible_flag, new_label, bound = self.check_two_state_feasible(left_state, right_state, left_TA, right_TA)
                    if feasible_flag:  # only non-empty region can define new state
                        new_node = (left_state, right_state)
                        self.add_node(new_node)
                        self.vertices_to_regions_label[new_node] = new_label
                        self.product_time_bound(bound, new_node, left_TA, right_TA)
                        self.recompute_state_time_bound(new_node)
                        self.activate_time_var[new_node] = left_TA.activate_time_var[left_state].union(right_TA.activate_time_var[right_state])
                        if left_state in left_TA.accepting_vertices and right_state in right_TA.accepting_vertices:
                            self.accepting_vertices.append(new_node)
                else:
                    feasible_flag = True

        return feasible_flag

    def product_time_bound(self, new_bound, new_state, left_TA, right_TA):

        new_state_time_bound = dict()

        for time_var in self.time_variables:
            if time_var == 'c0':
                new_state_time_bound[time_var] = new_bound
            elif time_var in left_TA.time_variables:
                bound = left_TA.state_time_bound[new_state[0]][time_var]
                new_state_time_bound[time_var] = (bound[0], min(bound[1], new_bound[1]))
            else:
                bound = right_TA.state_time_bound[new_state[1]][time_var]
                new_state_time_bound[time_var] = (bound[0], min(bound[1], new_bound[1]))

        self.state_time_bound[new_state] = new_state_time_bound

    def Product_of_TA(self, other):

        # This product construction intentionally explores only acyclic paths
        # in the discrete TA state graph. Once a candidate next_node already
        # appears in now_path, we record the edge but do not continue unfolding
        # from that revisit. This keeps the construction finite and predictable,
        # but means loop-like timed behaviors must be modeled by bounded
        # template unfolding (e.g. repeat_number in GF/GFL) rather than by
        # unrestricted revisits during product exploration.
        new_TA = Time_Automaton()
        new_TA.time_variables = self.time_variables.union(other.time_variables)
        new_TA.feasible_label = list(self.feasible_label)
        new_TA.end_time = self.end_time
        new_TA.template_metadata = (
            [dict(item) for item in self.template_metadata]
            + [dict(item) for item in other.template_metadata]
        )
        # compute the initial state
        new_TA.product_from_two_lists(self.initial_vertices, other.initial_vertices, self, other)
        new_TA.initial_vertices = list(new_TA.nodes)

        explore_list = [node for node in new_TA.initial_vertices]  # the node to unfold
        #length_list = [1 for num in range(len(explore_list))]
        path_list = [[explore_list[num]] for num in range(len(explore_list))]
        bound_list = [(new_TA.invariance_time_bound(self, 0, self.state_time_bound[explore_list[num][0]]),
                       new_TA.invariance_time_bound(other, 0, other.state_time_bound[explore_list[num][1]])) for num in range(len(explore_list))]
        time_constraint_dict = dict()
        for time_var in new_TA.time_variables:  #for time_var in new_TA.product_variable:
            time_constraint_dict[time_var] = (0, 0)
        explore_time_list = [time_constraint_dict.copy() for num in range(len(explore_list))]  # used to check feasibility of edge

        while len(explore_list) != 0:
            now_node = explore_list.pop()  # the node to search
            #now_length = length_list.pop()
            now_path = path_list.pop()
            now_time_cons = explore_time_list.pop()
            now_bound = bound_list.pop()

            for next_node_left in list(self.successors(now_node[0])):  # left transition
                feasible_flag_node = new_TA.product_from_two_lists([next_node_left], [now_node[1]], self, other)
                edge = (now_node[0], next_node_left)
                if feasible_flag_node:
                    next_node = (next_node_left, now_node[1])
                    next_time_cons, feasible_flag_edge = new_TA.check_edge_feasible(now_time_cons, self, edge, new_TA.state_time_bound[next_node], now_bound[1])

                    if feasible_flag_edge:  # the next node should feasible
                        #if now_length < len(self.nodes) + len(other.nodes):
                        if next_node not in now_path:
                            explore_list.append(next_node)
                            new_path = now_path.copy()
                            new_path.append(next_node)
                            path_list.append(new_path)
                            #length_list.append(now_length+1)
                            explore_time_list.append(next_time_cons)
                            new_bound = new_TA.invariance_time_bound(other, next_time_cons['c0'][1], new_TA.state_time_bound[next_node])
                            bound_list.append((new_bound, now_bound[1]))

                        new_TA.add_edge_in_TA(self, now_node, next_node, (now_node[0], next_node_left))

            for next_node_right in list(other.successors(now_node[1])):  # right transition
                feasible_flag_node = new_TA.product_from_two_lists([now_node[0]], [next_node_right], self, other)
                edge = (now_node[1], next_node_right)
                if feasible_flag_node:  # the next node should feasible
                    next_node = (now_node[0], next_node_right)
                    next_time_cons, feasible_flag_edge = new_TA.check_edge_feasible(now_time_cons, other, edge, new_TA.state_time_bound[next_node], now_bound[0])

                    if feasible_flag_edge:
                        #if now_length < len(self.nodes) + len(other.nodes):
                        if next_node not in now_path:
                            explore_list.append(next_node)
                            new_path = now_path.copy()
                            new_path.append(next_node)
                            path_list.append(new_path)
                            #length_list.append(now_length + 1)
                            explore_time_list.append(next_time_cons)
                            new_bound = new_TA.invariance_time_bound(self, next_time_cons['c0'][1],
                                                                     new_TA.state_time_bound[next_node])
                            bound_list.append((now_bound[0], new_bound))

                        new_TA.add_edge_in_TA(other, now_node, next_node, (now_node[1], next_node_right))

        new_TA.Prune()
        return new_TA

    def Union_of_TA(self, other):

        new_TA = Time_Automaton()
        new_TA.add_nodes_from(list(self.nodes)+list(other.nodes))
        new_TA.add_edges_from(list(self.edges)+list(other.edges))
        new_TA.initial_vertices = list(self.initial_vertices) + list(other.initial_vertices)
        new_TA.time_variables = self.time_variables.union(other.time_variables)
        new_TA.end_time = self.end_time
        new_TA.template_metadata = (
            [dict(item) for item in self.template_metadata]
            + [dict(item) for item in other.template_metadata]
        )

        for node in list(self.nodes)+list(other.nodes):

            if node in self.nodes:
                new_state_time_bound = new_TA.augment_time_constraint(new_TA.time_variables, self.time_variables, self.state_time_bound[node])
            else:
                new_state_time_bound = new_TA.augment_time_constraint(new_TA.time_variables, other.time_variables,
                                                                      other.state_time_bound[node])
            new_TA.state_time_bound[node] = new_state_time_bound
            new_TA.recompute_state_time_bound(node)

        for edge in list(self.edges)+list(other.edges):

            if edge in self.edges:
                new_edge_time_cons = new_TA.augment_time_constraint(new_TA.time_variables, self.time_variables, self.edges_time_constraint[edge])
            else:
                new_edge_time_cons = new_TA.augment_time_constraint(new_TA.time_variables, other.time_variables,
                                                                    other.edges_time_constraint[edge])

            new_TA.edges_time_constraint[edge] = new_edge_time_cons

        new_TA.edges_reset_variables = {
            edge: set(reset_vars) for edge, reset_vars in {**self.edges_reset_variables, **other.edges_reset_variables}.items()
        }
        new_TA.vertices_to_regions_label = {
            node: set(region_label) for node, region_label in {**self.vertices_to_regions_label, **other.vertices_to_regions_label}.items()
        }
        new_TA.activate_time_var = {
            node: set(active_vars) for node, active_vars in {**self.activate_time_var, **other.activate_time_var}.items()
        }
        new_TA.accepting_vertices = list(self.accepting_vertices) + list(other.accepting_vertices)
        new_TA.feasible_label = list(self.feasible_label)

        return new_TA

    def __and__(self, other):

        return self.Product_of_TA(other)

    def __or__(self, other):

        return self.Union_of_TA(other)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # # feasible_list = [set(['W']), set(['W', 'a']), set(['W', 'b']), set(['W', 'c']), set(['W', 'd']), set(['W', 'd', 'a']), set(['W', 'd', 'b']), set(['W', 'd', 'c'])]  # W is workspace, a,b region has intersection
    # # #TA_0 = Time_Automaton('G', [20, 0, 20], [set(['W']), set(['W'])], 0, feasible_list)
    # # TA_1 = Time_Automaton('U', [20, 0, 5], [set(['W']), set(['W']), set(['W', 'a'])], 1, feasible_list)
    # # #TA_2 = Time_Automaton('FG', [20, 9, 14, 1, 5], [set(['W']), set(['W', 'c'])], 2, feasible_list, 'c1')
    # # TA_2 = Time_Automaton('U', [20, 4, 8], [set(['W']), set(['W']), set(['b'])], 2, feasible_list)
    # # #TA_3 = Time_Automaton('GF', [20, 3, 8, 0, 3], [set(['W']), set(['W', 'b']), set(['W', 'd'])], 3, feasible_list, 'c2', repeat_number=2)
    # # TA_3 = Time_Automaton('U', [20, 10, 18], [set(['W']), set(['W']), set(['c'])], 3, feasible_list)
    # # # #TA = (TA_1 & TA_3)
    # # # start = time.time()
    # # # #TA = TA_1 & TA_2
    # # TA = ((TA_1 & TA_2) & TA_3) #& TA_0
    # # end = time.time()
    # # print(end-start)
    #
    # # feasible_list = [set(['W', 'b', 'd']), set(['W', 'a', 'b', 'd']),
    # #                  set(['W', 'a', 'd']), set(['W', 'c', 'd', 'b']), set(['W', 'b', 'c'])]
    # # TA_1 = Time_Automaton('GF', [15, 2, 7, 0.1, 2], [set(['W']), set(['a']), set(['b'])], 0, feasible_list,
    # #                       'c1', repeat_number=2)
    # #
    # # TA_2 = Time_Automaton('GF', [15, 2, 7, 0.1, 2], [set(['W']), set(['c']), set(['d'])], 1, feasible_list,
    # #                       'c2', repeat_number=2)
    # # start = time.time()
    # # TA = TA_1 & TA_2
    # # end = time.time()
    # # print(end-start)
    # # print(len(TA.nodes))
    # # print(len(TA.edges))
    #
    # # TA.re_name_time_variables()
    # # print(TA.time_variables)
    # # print(TA.edges_reset_variables)
    # # print(TA.edges_time_constraint)
    # # feasible_list = [{'a', 'W', 'e'}, {'W', 'd', 'e'}, {'W', 'd'}, {'b', 'W', 'd'}, {'W', 'e'}, {'W', 'd', 'e', 'c'}]
    # # TA_1 = Time_Automaton('GF', [10, 0.5, 5.5, 0, 2], [set(['W']), set(['W', 'a']), set(['W', 'd'])], 0, feasible_list, 'c1', 2)  # bU[0,4]a
    # # TA_2 = Time_Automaton('GF', [10, 0.5, 5.5, 0, 2], [set(['W']), set(['W', 'b']), set(['W', 'e'])], 1, feasible_list, 'c2', 2)  # dU[5,8]e
    # # TA_3 = Time_Automaton('FG', [10, 6.5, 7, 0, 2.5], [set(['W']), set(['W', 'c'])], 2, feasible_list, 'c3')  # F[0,10]f
    # # TA_1 = Time_Automaton('GF', [10, 0.5, 4, 0.1, 3], [set(['W']), set(['W', 'a']), set(['W', 'd'])], 0, feasible_list, 'c1', repeat_number=1)  # bU[0,4]a
    # # TA_2 = Time_Automaton('GF', [10, 0.5, 4, 0.1, 3], [set(['W']), set(['W', 'b']), set(['W', 'e'])], 1, feasible_list, 'c2', repeat_number=2)  # dU[5,8]e
    # # TA_3 = Time_Automaton('FG', [10, 7.5, 8, 0.1, 2], [set(['W']), set(['W', 'c'])], 2, feasible_list, 'c3')  # F[0,10]f
    # # TA_1 = Time_Automaton('U', [10, 4.5, 6.5], [set(['W']), set(['W']), set(['W', 'c'])], 0, feasible_list)
    # # TA_2 = Time_Automaton('U', [10, 1, 3], [set(['W']), set(['W']), set(['W', 'b'])], 1, feasible_list)
    # # TA_3 = Time_Automaton('U', [10, 7.5, 9], [set(['W']), set(['W']), set(['W', 'a'])], 2, feasible_list)
    # # TA_start_time = time.time()
    # # TA = (TA_1 & TA_2) & TA_3
    # # TA_time = time.time() - TA_start_time
    # # print(len(TA.nodes))
    # # print(len(TA.edges))
    # #TA = TA_1 & TA_3
    # # print(TA_time)
    # # print(TA.time_variables)
    # # print(TA.edges_reset_variables)
    # # print(TA.edges_time_constraint)
    # # print(TA.vertices_to_regions_label)
    # # for node in TA.nodes:
    # #     print(node, TA.state_time_bound[node])
    # pos = nx.spring_layout(TA)  # 布局算法
    # nx.draw(TA, pos, with_labels=True, node_color='skyblue', node_size=800,
    #         arrows=True, arrowsize=20, edge_color='gray')
    #
    # plt.title("Directed Graph Example")
    # plt.show()

    feasible_list = [{'W', 'l'}, {'n', 'W'}, {'o1', 'W', 'o'}, {'W', 'o4', 'o'}, {'o2', 'W', 'o'}, {'W', 'o', 'o3'}, {'t', 'W', 'n'}, {'n', 'c', 'W'}]
    TA_1 = Time_Automaton('seq', [30, 0.01, 29.99], [set(['W']), set(['W', 'o1']), set(['W', 'o2']), set(['W', 'o3']), set(['W', 'o4'])], 4, feasible_list)
    TA_6 = Time_Automaton('res', [30, 29.9, 10.], [set(['W']), set(['W', 'o']), set(['W', 'l']), set(['W', 'n']), set(['W', 't'])], 6, feasible_list, 'c2')
    TA = TA_1 & TA_6  # & TA_5
    print(len(TA.nodes))
