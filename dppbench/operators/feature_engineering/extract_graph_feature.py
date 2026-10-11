from ..base_op import TabularOp


class ExtractGraphFeature(TabularOp):
    """Extract simple graph structural features from an edge table."""

    PREFIX = "graph_"

    def __init__(self, source_col="src", target_col="dst", features=None,
                 directed=False):
        super().__init__(name="ExtractGraphFeature")
        self.source_col = source_col
        self.target_col = target_col
        self.features = features or ["degree", "pagerank"]
        self.directed = bool(directed)

    def get_op_description(self):
        description = """Operator name: ExtractGraphFeature

Function description:
Build a graph from an edge table and append node-level
degree/PageRank/community-like features for source and target nodes.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'src': ['u1', 'u1', 'u2'], 'dst': ['i1', 'i2', 'i2']})
>>> op = ExtractGraphFeature(source_col='src', target_col='dst', features=['degree'])
>>> op.transform(df)
  src dst  graph_src_degree  graph_dst_degree
0  u1  i1                 2                 1
1  u1  i2                 2                 2
2  u2  i2                 1                 2

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: ExtractGraphFeature
    prev:
    - s0
    params:
      source_col: user_id
      target_col: item_id
      features:
      - degree
      - pagerank
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        if self.source_col not in df.columns or self.target_col not in df.columns:
            return df
        df = df.copy()
        try:
            import networkx as nx
            graph = nx.DiGraph() if self.directed else nx.Graph()
            graph.add_edges_from(df[[self.source_col, self.target_col]].dropna().itertuples(index=False, name=None))
            degree = dict(graph.degree()) if "degree" in self.features else {}
            pagerank = nx.pagerank(graph) if "pagerank" in self.features and len(graph) else {}
        except Exception as exc:
            print(f"  [ExtractGraphFeature] networkx unavailable, using counts: {exc}")
            src_counts = df[self.source_col].value_counts()
            dst_counts = df[self.target_col].value_counts()
            degree = (src_counts.add(dst_counts, fill_value=0)).to_dict()
            pagerank = {}
        for node_col in (self.source_col, self.target_col):
            if "degree" in self.features:
                df[f"{self.PREFIX}{node_col}_degree"] = df[node_col].map(degree).fillna(0)
            if "pagerank" in self.features:
                df[f"{self.PREFIX}{node_col}_pagerank"] = df[node_col].map(pagerank).fillna(0.0)
        return df
