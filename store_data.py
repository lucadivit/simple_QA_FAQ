from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

faq_dict = {
    1: {
        "questions": [
            "Quali sono le politiche di reso e cambio per le scarpe acquistate online?",
            "Quali sono i termini per il reso e la sostituzione delle scarpe acquistate tramite il vostro sito web?",
            "Cosa prevedono le vostre regole per il reso e il cambio di scarpe acquistate online?",
            "Posso restituire o cambiare le scarpe acquistate online? Quali sono le condizioni?"
        ],
        "answer": "Le politiche di reso e cambio prevedono un periodo di X giorni dalla data di acquisto durante il quale è possibile restituire o cambiare le scarpe in condizioni non indossate e con l'imballaggio originale. Per ulteriori dettagli, consultare la nostra pagina delle politiche di reso sul sito."
    },
    2: {
        "questions": [
            "Come posso determinare la giusta taglia quando compro le scarpe dal vostro negozio online?",
            "Qual è il metodo consigliato per scegliere la taglia corretta durante l'acquisto di scarpe dal vostro sito?",
            "Cosa dovrei fare per assicurarmi di ordinare la taglia giusta quando faccio acquisti nel vostro negozio online?",
            "Ci sono suggerimenti per individuare la taglia corretta quando ordino scarpe dal vostro negozio virtuale?"
        ],
        "answer": "Per determinare la giusta taglia, ti consigliamo di seguire la nostra guida alle taglie sul sito. Puoi anche contattare il nostro servizio clienti per assistenza personalizzata nella scelta della taglia corretta per il tuo tipo di piede."
    },
    3: {
        "questions": [
            "Quali modalità di pagamento accettate nel vostro negozio di calzature?",
            "Quali tipi di pagamento sono validi per gli acquisti nel vostro negozio di scarpe?",
            "Accettate diverse forme di pagamento nel vostro negozio di calzature?",
            "Quali sono le opzioni di pagamento disponibili quando acquisto scarpe da voi?"
        ],
        "answer": "Accettiamo pagamenti con carta di credito, bonifico bancario e PayPal. Tutte le transazioni sono sicure e protette. Per maggiori dettagli, consulta la sezione Pagamenti sul nostro sito."
    },
    4: {
        "questions": [
            "Le vostre scarpe sono coperte da una garanzia? Quali sono i dettagli?",
            "Offrite una garanzia per le vostre calzature? Quali sono i particolari da conoscere?",
            "Cosa include la garanzia per le vostre scarpe? Quali sono i termini da considerare?",
            "Posso contare su una garanzia per le scarpe che acquisto dal vostro negozio? Quali sono i dettagli?"
        ],
        "answer": "Tutte le nostre scarpe sono coperte da una garanzia di qualità. La garanzia copre difetti di produzione e materiali. Per attivare la garanzia, conserva lo scontrino di acquisto e contattaci entro X giorni dalla data di acquisto."
    },
    5: {
        "questions": [
            "Offrite la spedizione internazionale? Quali sono i costi e i tempi di consegna?",
            "Posso ricevere le vostre scarpe all'estero? Quali sono i tempi di spedizione e i costi associati?",
            "È possibile effettuare ordini internazionali nel vostro negozio di scarpe? Quali sono i costi di spedizione?",
            "Quali sono i dettagli sulla spedizione internazionale per le scarpe acquistate dal vostro negozio?"
        ],
        "answer": "Offriamo la spedizione in territorio nazionale ed internazionale. I costi e i tempi di consegna variano a seconda della destinazione. Per conoscere i dettagli specifici, consulta la nostra pagina di spedizione sul sito."
    }
}

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
client = QdrantClient(path="qdrant_db")
vectors = VectorParams(size=512, distance=Distance.COSINE)
collection_name = "shoes_shop_FAQ"


def initialize_collection():
    # client.delete_collection(collection_name)
    try:
        client.create_collection(collection_name, vectors)
        print(f"Collection {collection_name} Created")
        inc_id = 0
        points = []
        for key in faq_dict:
            sentences = faq_dict[key]["questions"]
            answer = faq_dict[key]["answer"]
            for sentence in sentences:
                sentence = sentence.lower()
                sentence_embedding = model.encode(sentence, convert_to_numpy=True).tolist()
                point = PointStruct(id=inc_id, vector=sentence_embedding, payload={"answer": answer})
                points.append(point)
                inc_id += 1
        client.upsert(collection_name=collection_name, points=points)
        print(f"Collection {collection_name} Populated")
    except:
        print(f"Collection {collection_name} Already Exists")


def do_a_question(sentence: str) -> str:
    sentence = sentence.lower()
    sentence_embedding = model.encode(sentence, convert_to_numpy=True).tolist()
    search_result = client.search(collection_name=collection_name, query_vector=sentence_embedding, limit=1, score_threshold=0.4)
    if len(search_result) == 1:
        answer = search_result[0].payload["answer"]
    else:
        answer = "Non ho capito la tua domanda, prova a formularla diversamente!"
    return answer
