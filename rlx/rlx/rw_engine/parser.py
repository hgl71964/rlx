from typing import Optional, Any

from rlx.utils.common import get_logger
from rlx.frontend.graph import Graph, Node, Edge
from rlx.frontend.registry import get_node_type

logger = get_logger(__name__)


class rlx_Edge(Edge):
    def __init__(self, idx, attr, edge_type, trace):
        self.idx = idx
        self.attr = attr
        # can be only Const type or Symbol type
        self.edge_type = edge_type
        self.uses = []
        self.trace = trace

    def get_type(self):
        return self.edge_type

    def get_attr(self):
        return self.attr

    def get_uses(self):
        return self.uses

    def get_trace(self):
        return self.trace


class rlx_Node(Node):
    def __init__(self, idx, attr, node_type, inputs):
        self.idx = idx
        self.attr = attr
        self.node_type = node_type
        self.inputs = inputs
        self.outputs = []

        if inputs is not None:
            assert isinstance(inputs, list), f"{type(inputs)}??"
            for inp in inputs:
                assert isinstance(inp, rlx_Edge)
                inp.uses.append(self)

    def get_type(self):
        return self.node_type

    def get_attr(self):
        return self.attr

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs


class rlx_Graph(Graph):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class Parser:
    def __init__(self, graph: Graph):
        edges = graph.get_edges()
        nodes = graph.get_nodes()

        # build internal graph
        edge_map = {}
        for i, e in enumerate(edges):
            ee = rlx_Edge(i, e.get_attr(), e.get_type(), None)
            edge_map[e] = ee

        node_map = {}
        for i, n in enumerate(nodes):
            nn = rlx_Node(i, n.get_attr(), n.get_type(), None)
            node_map[n] = nn

        # add connection
        self.edges = []
        for i, e in enumerate(edges):
            ee = edge_map[e]
            e_trace = e.get_trace()
            if e_trace is not None:
                assert e_trace in node_map, "Graph may be broken, are all NODEs added?"
                ee_trace = node_map[e_trace]
                ee.trace = ee_trace

            # TODO add idx too? so that it helps
            # if a edge is input twice to a node? (a - a = 0)
            e_uses = e.get_uses()
            for e_use in e_uses:
                assert e_use in node_map, "Graph may be broken, are all NODEs added?"
                ee_use = node_map[e_use]
                ee.uses.append(ee_use)

            self.edges.append(ee)  # add to self

        self.nodes = []
        for i, n in enumerate(nodes):
            nn = node_map[n]
            nn.inputs = []

            for inp in n.get_inputs():
                assert inp in edge_map, "Graph may be broken, are all EDGES added?"
                ee = edge_map[inp]
                nn.inputs.append(ee)

            for out in n.get_outputs():
                assert out in edge_map, "Graph may be broken, are all EDGES added?"
                ee = edge_map[out]
                nn.outputs.append(ee)

            self.nodes.append(nn)

        self._detect_circle([e for e in self.edges if len(e.uses) == 0])
        self._check_uses(self.edges)
        self._infer_edge_type()

    def _detect_circle(self, output_edges: list[rlx_Edge]):
        visited, path = set(), set()
        stack = []
        node_cnt, edge_cnt = 0, 0

        def dfs(obj):
            nonlocal node_cnt, edge_cnt
            if obj is None or obj in visited:
                return

            if isinstance(obj, rlx_Node):
                if obj in path:
                    logger.critical(f"circle detected! {obj.idx}")
                    logger.critical(node_cnt, edge_cnt)
                    logger.critical(stack)
                    raise RuntimeError("circle detected")
                path.add(obj)
                stack.append(obj.idx)
                for inp in obj.inputs:
                    dfs(inp)
                node_cnt += 1
                visited.add(obj)
                path.remove(obj)
                stack.pop()

            if isinstance(obj, rlx_Edge):
                if obj in path:
                    logger.critical(f"circle detected! {obj.idx}")
                    logger.critical(node_cnt, edge_cnt)
                    logger.critical(stack)
                    raise RuntimeError("circle detected")
                path.add(obj)
                stack.append(obj.idx)
                dfs(obj.trace)
                edge_cnt += 1
                visited.add(obj)
                path.remove(obj)
                stack.pop()

        try:
            for out in output_edges:
                dfs(out)
        except Exception as e:
            logger.critical("[PARSER] circle detected!")
            raise RuntimeError("circle detected")

    def _check_uses(self, edges):
        sinks = []
        for e in edges:
            if len(e.uses) == 0:
                sinks.append(e.idx)
        logger.info(f"graph sinks: {sinks}")

    def _infer_edge_type(self):
        # A rlx_Edge should be either Const or Var
        # but constant folding is supposed to be done by users
        visited = set()
        _, const_type, var_type = get_node_type()

        def dfs(obj):
            if obj is None or obj in visited:
                return

            if isinstance(obj, rlx_Node):
                for inp in obj.inputs:
                    dfs(inp)

                var = False
                for inp in obj.inputs:
                    if inp.edge_type is None:
                        raise RuntimeError(
                            "forget to set leaf node? initial edge type inference fails"
                        )

                    if inp.edge_type == var_type:
                        var = True

                out_type = var_type if var else const_type
                for out in obj.outputs:
                    out.edge_type = out_type

                visited.add(obj)

            if isinstance(obj, rlx_Edge):
                dfs(obj.trace)
                visited.add(obj)

        # need to dfs from topological order!
        output_edge = []
        for e in self.edges:
            if len(e.uses) == 0:
                output_edge.append(e)

        for e in output_edge:
            dfs(e)

    def _build_mapping(self):
        self.edge_map = {}
        self.node_map = {}
        for e in self.edges:
            self.edge_map[e.idx] = e
        for n in self.nodes:
            self.node_map[n.idx] = n

    def viz(self, edges, path="rlx_graph_viz", check=True):
        if check:
            self._check(edges)

        # plot for debugging
        import graphviz
        g = graphviz.Digraph("rlx_Graph", format="pdf", filename=f"{path}")

        built = set()
        for i, e in enumerate(edges):
            assert isinstance(
                e,
                rlx_Edge), f"viz only works for rlx_Graph, but got {type(e)}"
            eid = e.idx
            etype = e.get_type().name
            if isinstance(e.attr, int):
                e_label = f"Edge-{eid}_{etype}({e.attr})"
            else:
                e_label = f"Edge-{eid}_{etype}"
            g.node(e_label, e_label, **{"shape": "rectangle"})

            if e.trace is not None:
                nid = e.trace.idx
                ntype = e.trace.get_type().name
                label = f"Node-{nid}_{ntype}"
                if nid not in built:
                    g.node(label, label, **{"shape": "circle"})
                    built.add(nid)
                g.edge(label, e_label)  # trace -> edge

            for n in e.uses:
                nid = n.idx
                ntype = n.get_type().name
                label = f"Node-{nid}_{ntype}"
                if nid not in built:
                    g.node(label, label, **{"shape": "circle"})
                    built.add(nid)
                g.edge(e_label, label)  # edge -> uses

        g.render()

    def _check(self, edges):
        for i, e in enumerate(edges):
            ok = True
            # input node
            if e.trace is not None:
                n = e.trace
                if e not in n.outputs:
                    ok = False
                    logger.critical(
                        f"An exception occurred: {e.idx}, {e.edge_type}")
                    logger.critical(f"{n.idx}, {n.node_type}")
                    for out in n.outputs:
                        logger.critical(f"{out.idx}, {out.edge_type}")

                ok = ok and self._check_node(n)

            # output node
            for n in e.uses:
                if e not in n.inputs:
                    ok = False
                    logger.critical(
                        f"An exception occurred: {e.idx}, {e.edge_type}")
                    logger.critical(f"{n.idx}, {n.node_type}")
                    for inp in n.inputs:
                        logger.critical(f"{inp.idx}, {inp.edge_type}")

                ok = ok and self._check_node(n)

            if not ok:
                raise Exception

    def _check_node(self, n):
        ok = True
        if len(n.inputs) > 2 or len(n.inputs) < 1:
            ok = False
            logger.erorr(f"{n.idx}, {n.node_type}, {len(n.inputs)}")
            for inp in n.inputs:
                logger.critical(f"{inp.idx}, {inp.edge_type}")
        if len(n.outputs) != 1:
            ok = False
            logger.error(f"{n.idx}, {n.node_type}, {len(n.outputs)}")
            for out in n.outputs:
                logger.critical(f"{out.idx}, {out.edge_type}")
        return ok
