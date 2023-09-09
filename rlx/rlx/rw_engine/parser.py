from typing import Optional, Any

from rlx.utils.common import get_logger
from rlx.frontend.graph import Graph, Node, Edge
from rlx.frontend.registry import get_node_type

logger = get_logger(__name__)


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
        self.edges = graph.get_edges()
        self.nodes = graph.get_nodes()

        # run some checks
        output_edges = [e for e in self.edges if len(e.get_uses()) == 0]
        self._detect_circle(output_edges)
        self._check_uses(self.edges)
        self._infer_edge_type(output_edges)

    def _detect_circle(self, output_edges: list[Edge]):
        visited, path = set(), set()
        stack = []
        node_cnt, edge_cnt = 0, 0

        def dfs(obj):
            nonlocal node_cnt, edge_cnt
            if obj is None or obj in visited:
                return

            if isinstance(obj, Node):
                if obj in path:
                    logger.critical(f"circle detected! {obj.get_idx()}")
                    logger.critical(node_cnt, edge_cnt)
                    logger.critical(stack)
                    raise RuntimeError("circle detected")
                path.add(obj)
                stack.append(obj.get_idx())
                for inp in obj.get_inputs():
                    dfs(inp)
                node_cnt += 1
                visited.add(obj)
                path.remove(obj)
                stack.pop()

            if isinstance(obj, Edge):
                if obj in path:
                    logger.critical(f"circle detected! {obj.get_idx()}")
                    logger.critical(node_cnt, edge_cnt)
                    logger.critical(stack)
                    raise RuntimeError("circle detected")
                path.add(obj)
                stack.append(obj.get_idx())
                dfs(obj.get_trace())
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
            if len(e.get_uses()) == 0:
                sinks.append(e.get_idx())
        logger.info(f"graph sinks: {sinks}")

    def _infer_edge_type(self, output_edges: list[Edge]):
        # A rlx_Edge should be either Const or Var
        # but constant folding is supposed to be done by users
        visited = set()
        _, const_type, var_type = get_node_type()
        node_cnt, edge_cnt = 0, 0

        def dfs(obj):
            nonlocal node_cnt, edge_cnt
            if obj is None or obj in visited:
                return

            if isinstance(obj, Node):
                for inp in obj.get_inputs():
                    dfs(inp)

                var = False
                for inp in obj.get_inputs():
                    if inp.get_type() is None:
                        raise RuntimeError(
                            "forget to set leaf node? initial edge type inference fails"
                        )

                    if inp.get_type() == var_type:
                        var = True

                out_type = var_type if var else const_type
                for out in obj.get_outputs():
                    out.set_type(out_type)

                if hasattr(obj, "_rlx_idx"):
                    raise RuntimeError(
                        f"user should not set {obj.get_idx()} _rlx_idx {obj._rlx_idx}"
                    )
                obj._rlx_idx = node_cnt  # attach internal idx
                node_cnt += 1
                visited.add(obj)

            if isinstance(obj, Edge):
                dfs(obj.get_trace())
                if hasattr(obj, "_rlx_idx"):
                    raise RuntimeError(
                        f"user should not set {obj.get_idx()} _rlx_idx {obj._rlx_idx}"
                    )
                obj._rlx_idx = edge_cnt  # attach internal idx
                edge_cnt += 1
                visited.add(obj)

        # need to dfs from topological order
        for e in output_edges:
            dfs(e)

    def _build_mapping(self):
        self.edge_map = {}
        self.node_map = {}
        for e in self.edges:
            self.edge_map[e.get_idx()] = e
        for n in self.nodes:
            self.node_map[n.get_idx()] = n

    def viz(self, edges, path="rlx_graph_viz", check=True):
        if check:
            self._check(edges)

        # plot for debugging
        import graphviz  # type: ignore
        g = graphviz.Digraph("rlx_Graph", format="pdf", filename=f"{path}")

        built = set()
        for _, e in enumerate(edges):
            assert isinstance(
                e, Edge), f"viz only works for rlx_Graph, but got {type(e)}"
            eid = e.get_idx()
            etype = e.get_type().name
            if isinstance(e.get_attr(), int):
                e_label = f"Edge-{eid}_{etype}({e.get_attr()})"
            else:
                e_label = f"Edge-{eid}_{etype}"
            g.node(e_label, e_label, **{"shape": "rectangle"})

            if e.get_trace() is not None:
                nid = e.get_trace().get_idx()
                ntype = e.get_trace().get_type().name
                label = f"Node-{nid}_{ntype}"
                if nid not in built:
                    g.node(label, label, **{"shape": "circle"})
                    built.add(nid)
                g.edge(label, e_label)  # trace -> edge

            for n in e.get_uses():
                nid = n.get_idx()
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
            if e.get_trace() is not None:
                n = e.get_trace()
                if e not in n.get_outputs():
                    ok = False
                    logger.critical(
                        f"An exception occurred: {e.get_idx()}, {e.get_type()}"
                    )
                    logger.critical(f"{n.get_idx()}, {n.get_type()}")
                    for out in n.outputs:
                        logger.critical(f"{out.get_idx()}, {out.get_type()}")

                ok = ok and self._check_node(n)

            # output node
            for n in e.get_uses():
                if e not in n.get_inputs():
                    ok = False
                    logger.critical(
                        f"An exception occurred: {e.get_idx()}, {e.get_type()}"
                    )
                    logger.critical(f"{n.get_idx()}, {n.get_type()}")
                    for inp in n.get_inputs():
                        logger.critical(f"{inp.get_idx()}, {inp.get_type()}")

                ok = ok and self._check_node(n)

            if not ok:
                raise Exception

    def _check_node(self, n):
        ok = True
        if len(n.get_inputs()) > 2 or len(n.get_inputs()) < 1:
            ok = False
            logger.erorr(
                f"{n.get_idx()}, {n.get_type()}, {len(n.get_inputs())}")
            for inp in n.get_inputs():
                logger.critical(f"{inp.get_idx()}, {inp.get_type()}")
        if len(n.get_outputs()) != 1:
            ok = False
            logger.error(
                f"{n.get_idx()}, {n.get_type()}, {len(n.get_outputs())}")
            for out in n.get_outputs():
                logger.critical(f"{out.get_idx()}, {out.get_type()}")
        return ok
