"""LanceDB vector store."""
from typing import Any, Dict, List, Optional, cast

# import numpy as np

from gpt_index.data_structs.node_v2 import DocumentRelationship, Node
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
    VectorStoreQuery,
)


class LanceDBVectorStore(VectorStore):
    """The LanceDB Vector Store

    The embeddings are stored in LanceDB.

    During query time,

    """
    stores_text = True

    def __init__(
        self,
        uri: str,
        table_name: str = "vectors",
        nprobes: int = 20,
        refine_factor: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        import_err_msg = (
            "`lancedb` package not found, please run `pip install lancedb`"
        )
        try:
            import lancedb  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        self.connection = lancedb.connect(uri)
        self.uri = uri
        self.table_name = table_name
        self.nprobes = nprobes
        self.refine_factor = refine_factor

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorStore":
        return cls(**config_dict)

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return {
            "uri": self.uri,
            "table_name": self.table_name,
            "nprobes": self.nprobes,
            "refine_factor": self.refine_factor,
        }

    def add(
            self,
            embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        data = []
        ids = []
        for result in embedding_results:
            data.append({
                "id": result.id,
                "doc_id": result.doc_id,
                "vector": result.embedding,
                "text": result.node.get_text(),
            })
            ids.append(result.id)

        if "vectors" in self.connection.table_names():
            tbl = self.connection.open_table("vectors")
            tbl.add(data)
        else:
            self.connection.create_table(self.table_name, data)
        return ids

    def query(
            self,
            query: VectorStoreQuery,
    ) -> VectorStoreQueryResult:
        table = self.connection.open_table(self.table_name)
        query = table.search(query.query_embedding) \
            .limit(query.similarity_top_k) \
            .nprobes(self.nprobes)

        if self.refine_factor is not None:
            query.refine_factor(self.refine_factor)

        results = query.to_df()
        nodes = []
        for _, item in results.iterrows():
            node = Node(
                doc_id=item.id,
                text=item.text,
                relationships={
                    DocumentRelationship.SOURCE: item.doc_id,
                }
            )
            nodes.append(node)

        return VectorStoreQueryResult(nodes=nodes, similarities=results["score"].tolist(), ids=results["id"].tolist())
