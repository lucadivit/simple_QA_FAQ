from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Creiamo QDrant in locale, ma possiamo fornire un indirizzo per istanza remota.
client = QdrantClient(":memory:")
collection_name = "test_collection"
# Per lavorare con i testi la distanza più adatta è la Coseno, ma ce ne sono altre
# DOT, EUCLID
vectors = VectorParams(size=4, distance=Distance.COSINE)
# Creiamo una collection
client.recreate_collection(collection_name=collection_name,
                           vectors_config=vectors)

# Inseriamo i dati nella collection
operation_info = client.upsert(collection_name=collection_name,
                               points=[
                                   PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"type": "test_1"}),
                                   PointStruct(id=2, vector=[0.3, 0.7, 0.8, 1.1], payload={"type": "test_1"}),
                                   PointStruct(id=3, vector=[1.3, 2.7, -0.8, -1.1], payload={"type": "test_1"}),
                                   PointStruct(id=4, vector=[1.3, 2.2, -1.8, 4.1], payload={"type": "test_2"}),
                                   PointStruct(id=5, vector=[1.9, -2.7, 1.2, 7.1], payload={"type": "test_2"}),
                                   PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"type": "test_2"})
                               ]
                               )

# Esempio di query
query = [0.2, 0.1, 0.9, 0.7]
search_result = client.search(collection_name=collection_name,
                              query_vector=query,
                              limit=3)
print("\nEsempio di ricerca:")
for point in search_result:
    print(point)

# Query con filtro
filter = Filter(must=[FieldCondition(key="type", match=MatchValue(value="test_2"))])
search_result = client.search(collection_name=collection_name,
                              query_vector=query,
                              query_filter=filter,
                              with_payload=True,
                              limit=3)

print("\nEsempio di ricerca con filtro:")
for point in search_result:
    print(point)

# Query con filtro e threshold
filter = Filter(must=[FieldCondition(key="type", match=MatchValue(value="test_2"))])
search_result = client.search(collection_name=collection_name,
                              query_vector=query,
                              query_filter=filter,
                              with_payload=True,
                              score_threshold=0.3,
                              limit=3)

print("\nEsempio di ricerca con filtro e threshold:")
for point in search_result:
    print(point)

# Recommendation: ci piace il 2 ma non ci piace l'1
search_result = client.recommend(collection_name=collection_name,
                                 positive=[2],
                                 negative=[1],
                                 limit=2)

print("\nEsempio di recommendation:")
for point in search_result:
    print(point)

# Altri metodi utili: retrieve -> ritorna un vettore dato un id, count -> conta gli elementi in una collection, delete -> cancella un elemento