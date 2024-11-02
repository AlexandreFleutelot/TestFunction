from rag_services.embeddings import EmbeddingService
from rag_services.chunking import create_chunks
from rag_services.storage import ChunkStore
from rag_services.models import Chunk


import azure.functions as func

import os
import logging
import json

#from dotenv import load_dotenv
#load_dotenv()

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Initialize services lazily to avoid cold start issues
embedding_service = None
chunk_store = None

@app.route(route="ingest_txt", methods=["POST"])
def ingest_txt(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to ingest text.')

    try:
        req_body = req.get_json()
        text = req_body.get('text')
        window_size = req_body.get('window_size', 6000)
        overlap_size = req_body.get('overlap_size', 3000)
        min_chunk_size = req_body.get('min_chunk_size', 100)
        max_chunk_size = req_body.get('max_chunk_size', 500)

        if not text:
            return func.HttpResponse("The 'text' field is required.", status_code=400)

        # Get embedding service (lazy initialization)
        embedding_service = get_embedding_services()

        # Get chunk store (lazy initialization)
        chunk_store = get_chunk_store()

        # Get embeddings and chunk the document
        token_embeddings, offset_mapping = embedding_service.get_token_embeddings(text)

        # Get optimal boundaries for splitting chunks
        chunks: List[Chunk] = create_chunks(
            text=text,
            token_embeddings=token_embeddings,
            offset_mapping=offset_mapping,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )

        return func.HttpResponse(json.dumps({"chunks": [chunk.to_dict() for chunk in chunks]}), status_code=200)

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return func.HttpResponse("Internal server error", status_code=500)

@app.route(route="retrieve_chunks", methods=["GET"])
def retrieve_chunks(req: func.HttpRequest) -> func.HttpResponse:
    try:
        embedding_service = get_embedding_services()
        chunk_store = get_chunk_store()
        
        query = req.params.get('query')
        top_k = req.params.get('top_k', 5)

        if not query:
            return func.HttpResponse(
                "Please provide a query in the request body",
                status_code=400
            )

        # Get query embedding
        token_embeddings, _ = embedding_service.get_token_embeddings(query)
        query_embedding = token_embeddings.mean(dim=0).numpy()

        # Search for similar chunks
        similar_chunks = chunk_store.search_similar_chunks(query_embedding, top_k)
        
        # Convert chunks to JSON-serializable format
        response_chunks = [chunk.to_dict() for chunk in similar_chunks]

        return func.HttpResponse(
            json.dumps({"chunks": response_chunks}),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error retrieving chunks: {str(e)}")
        return func.HttpResponse("An error occurred while retrieving the chunks.", status_code=500)
    
def get_embedding_services():
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service

def get_chunk_store():
    global chunk_store
    if chunk_store is None:
        chunk_store = ChunkStore(
            connection_string=os.getenv("COSMOS_CONNECTION_STRING"),
            database_name=os.getenv("COSMOS_DATABASE_NAME"),
            container_name=os.getenv("COSMOS_CONTAINER_NAME")
        )
    return chunk_store