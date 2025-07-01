from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct, Record
from typing import Dict, List, Optional, Union
import uuid
from ai import MAX_EMBED_CHUNK_SIZE, GeminiClient
from sheet import ProviderItem, ProviderWorkbook
from type import ItemReference, Provider, SheetName


QDRANT_COLLECTION_NAME = "HomeX"

class VectorStore():
    def __init__(self, client: QdrantClient):
        self.__client = client

    def get_record(self, item: ItemReference, with_vectors: bool = False) -> Optional[Record]:
        filter_condition = Filter(
            must=[
                FieldCondition(key="provider", match=MatchValue(value=item.provider.value)),
                FieldCondition(key="id", match=MatchValue(value=item.id))
            ]
        )
        matches, _ = self.__client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=filter_condition,
            limit=1,
            with_vectors=with_vectors
        )
        return matches[0] if matches else None


    def fetch_embedding(self, item: ItemReference) -> list[float]:
        record = self.get_record(item, True)
        if not record:
            raise Exception(f"No embedding record found for item ({item.provider} {item.id})")
        if not record.vector:
            raise Exception(f"No vector found in embedding record for item ({item.provider} {item.id})")
        return record.vector


    def get_records(self, items: List[ItemReference], with_vectors: bool = False) -> Dict[ItemReference, Record]:
        records, _ = self.__client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                should=[
                    Filter(
                        must=[
                            FieldCondition(key="provider", match=MatchValue(value=item.provider.value)),
                            FieldCondition(key="id", match=MatchValue(value=item.id)),
                        ]
                    )
                    for item in items
                ]
            ),
            limit=len(items),
            with_vectors=with_vectors
        )

        result: Dict[ItemReference, Record] = {}
        for record in records:
            item_ref = payload_to_item_ref(record.payload)
            result[item_ref] = record
        
        return result


    def store_embeddings(self, ai_client: GeminiClient, workbooks: Dict[Provider, ProviderWorkbook]):
        for workbook in workbooks.values():
            print(f"Processing embeddings for provider: {workbook.provider}")

            item_chunks = chunk_list(workbook.item_refs, MAX_EMBED_CHUNK_SIZE)
            for idx, item_chunk in enumerate(item_chunks):
                print(f"Embedding chunk {idx + 1} of {len(item_chunks)} ({workbook.provider})")

                items_to_embed: List[ProviderItem] = []
                embed_records = self.get_records(item_chunk, True)
                for item_ref in item_chunk:
                    if item_ref in embed_records:
                        continue

                    item = workbook.get_item_by_ref(item_ref)
                    items_to_embed.append(item)

                if not items_to_embed:
                    continue

                chunk_embeddings = ai_client.embed_chunk(items_to_embed)
                chunk_points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embed_result.embedding,
                        payload={
                            "provider": embed_result.item.provider.value,
                            "id": embed_result.item.id,
                            "sheet_name": embed_result.item.sheet_name.value
                        }
                    )
                    for embed_result in chunk_embeddings
                ]
                self.__client.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=chunk_points
                )
        
        print("Finished processing embeddings")


    def get_relevant_items(self, query: ItemReference, filters: List[Union[Provider, ItemReference]], limit: int = 1500):
        embedding = self.fetch_embedding(query)
        
        # Build filter conditions based on filter type
        filter_conditions = []
        for filter_item in filters:
            if isinstance(filter_item, Provider):
                # Filter by provider only
                filter_conditions.append(Filter(
                    must=[
                        FieldCondition(key="provider", match=MatchValue(value=filter_item.value))
                    ]
                ))
            elif isinstance(filter_item, ItemReference):
                # Filter by specific provider and ID
                filter_conditions.append(Filter(
                    must=[
                        FieldCondition(key="provider", match=MatchValue(value=filter_item.provider.value)),
                        FieldCondition(key="id", match=MatchValue(value=filter_item.id))
                    ]
                ))
        
        response = self.__client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=embedding,
            limit=limit,
            query_filter=Filter(should=filter_conditions)
        )

        return [
            payload_to_item_ref(record.payload)
            for record in response
        ]


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    

def payload_to_item_ref(payload: Optional[Dict]) -> ItemReference:
    if not payload:
        raise Exception("Missing payload")
    
    if not payload.get('provider'):
        raise Exception("Missing `provider` in payload")
    
    if not payload.get('sheet_name'):
        raise Exception("Missing `sheet_name` in payload")
    
    if not payload.get('id'):
        raise Exception("Missing `id` in payload")
    
    return ItemReference(
        provider=Provider(payload['provider']),
        sheet_name=SheetName(payload['sheet_name']),
        id=payload['id']
    )

