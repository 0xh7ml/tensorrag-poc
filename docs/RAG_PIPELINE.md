# RAG (Retrieval Augmented Generation) Pipeline

Build a complete RAG system using 6 individual cards. Each card handles one step, and data flows between them via S3 storage.

## Pipeline Overview

```
Document Loader → Text Chunker → Vector Embedder → Vector Store → RAG Query → Response Generator
```

## Project File Structure

Create the following folders and files in the **Editor** view:

```
rag-system/                   ← Project name
├── data/                     ← Folder
│   ├── document_loader.py    ← Card 1
│   └── text_chunker.py       ← Card 2
├── embedding/                ← Folder
│   ├── vector_embedder.py    ← Card 3
│   └── vector_store.py       ← Card 4  
├── retrieval/                ← Folder
│   └── rag_query.py          ← Card 5
└── generation/               ← Folder
    └── response_generator.py ← Card 6
```

## Card Connection Map

| # | Card | File | Folder | Receives from | Sends to |
|---|------|------|--------|--------------|----------|
| 1 | Document Loader | `document_loader.py` | `data/` | — (config: file paths) | `raw_documents` |
| 2 | Text Chunker | `text_chunker.py` | `data/` | `raw_documents` | `text_chunks` |
| 3 | Vector Embedder | `vector_embedder.py` | `embedding/` | `text_chunks` | `embeddings` |
| 4 | Vector Store | `vector_store.py` | `embedding/` | `embeddings` | `vector_db` |
| 5 | RAG Query | `rag_query.py` | `retrieval/` | `vector_db` | `retrieved_context` |
| 6 | Response Generator | `response_generator.py` | `generation/` | `retrieved_context` | `rag_response` |

---

## Card 1: Document Loader

**File:** `document_loader.py` | **Folder:** `data/`

Loads documents from various sources (PDF, text files, web pages).

```python
from cards.base import BaseCard
import requests
from pathlib import Path
import json
from urllib.parse import urlparse

class DocumentLoaderCard(BaseCard):
    card_type = "rag_document_loader"
    display_name = "Document Loader"
    description = "Load documents from various sources"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "source_type": {
            "type": "string",
            "label": "Source type (urls, text, sample)",
            "default": "sample"
        },
        "urls": {
            "type": "string",
            "label": "URLs (one per line)",
            "default": "https://en.wikipedia.org/wiki/Artificial_intelligence\nhttps://en.wikipedia.org/wiki/Machine_learning"
        },
        "text_content": {
            "type": "string", 
            "label": "Direct text input",
            "default": ""
        }
    }
    input_schema = {}
    output_schema = {"raw_documents": "json"}

    def execute(self, config, inputs, storage):
        source_type = config.get("source_type", "sample")
        
        documents = []
        
        if source_type == "sample":
            # Sample documents about AI/ML
            sample_docs = [
                {
                    "id": "doc_1",
                    "title": "Introduction to Machine Learning",
                    "content": """Machine learning is a subset of artificial intelligence that enables 
                    computers to learn and improve from experience without being explicitly programmed. 
                    It focuses on developing algorithms that can access data and use it to learn for themselves.
                    
                    There are three main types of machine learning: supervised learning, unsupervised learning, 
                    and reinforcement learning. Supervised learning uses labeled data to train models, 
                    unsupervised learning finds patterns in unlabeled data, and reinforcement learning 
                    learns through interaction with an environment.
                    
                    Common applications include image recognition, natural language processing, 
                    recommendation systems, and autonomous vehicles.""",
                    "source": "sample",
                    "metadata": {"category": "education", "length": "medium"}
                },
                {
                    "id": "doc_2", 
                    "title": "Deep Learning Fundamentals",
                    "content": """Deep learning is a specialized area of machine learning that uses 
                    neural networks with multiple layers to model and understand complex patterns. 
                    These networks are inspired by the structure and function of the human brain.
                    
                    Key concepts include neurons, layers, weights, biases, and activation functions. 
                    Popular architectures include Convolutional Neural Networks (CNNs) for image processing, 
                    Recurrent Neural Networks (RNNs) for sequential data, and Transformers for 
                    natural language understanding.
                    
                    Deep learning has achieved breakthrough results in computer vision, natural language 
                    processing, game playing, and scientific discovery. However, it requires large amounts 
                    of data and computational resources.""",
                    "source": "sample",
                    "metadata": {"category": "technical", "length": "medium"}
                },
                {
                    "id": "doc_3",
                    "title": "Ethics in AI Development", 
                    "content": """As artificial intelligence becomes more prevalent in society, 
                    ethical considerations become increasingly important. Key issues include bias 
                    in algorithms, privacy concerns, transparency and explainability, and the 
                    potential impact on employment.
                    
                    Bias can occur when training data is not representative or when algorithms 
                    inadvertently discriminate against certain groups. Privacy is threatened 
                    when AI systems collect and process personal data without proper safeguards.
                    
                    Transparency refers to understanding how AI systems make decisions, which is 
                    crucial for trust and accountability. The automation of jobs by AI raises 
                    concerns about unemployment and the need for workforce retraining.
                    
                    Developing ethical AI requires multidisciplinary collaboration between 
                    technologists, ethicists, policymakers, and society at large.""",
                    "source": "sample",
                    "metadata": {"category": "ethics", "length": "long"}
                }
            ]
            documents = sample_docs
            
        elif source_type == "urls":
            urls = [url.strip() for url in config.get("urls", "").split("\n") if url.strip()]
            
            for i, url in enumerate(urls):
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    
                    # Simple text extraction (in production, use libraries like BeautifulSoup)
                    content = response.text[:5000]  # Limit content length
                    
                    documents.append({
                        "id": f"url_{i+1}",
                        "title": f"Document from {urlparse(url).netloc}",
                        "content": content,
                        "source": url,
                        "metadata": {"type": "web", "url": url}
                    })
                except Exception as e:
                    documents.append({
                        "id": f"url_{i+1}_error", 
                        "title": f"Failed to load {url}",
                        "content": f"Error loading document: {str(e)}",
                        "source": url,
                        "metadata": {"type": "error", "url": url}
                    })
                    
        elif source_type == "text":
            text_content = config.get("text_content", "")
            if text_content.strip():
                documents.append({
                    "id": "text_input",
                    "title": "User-provided text",
                    "content": text_content,
                    "source": "user_input",
                    "metadata": {"type": "text_input"}
                })
        
        # Calculate statistics
        total_chars = sum(len(doc["content"]) for doc in documents)
        avg_length = total_chars / len(documents) if documents else 0
        
        doc_data = {
            "documents": documents,
            "total_documents": len(documents),
            "total_characters": total_chars,
            "average_length": avg_length,
            "source_type": source_type
        }
        
        ref = storage.save_json("_p", "_n", "raw_documents", doc_data)
        return {"raw_documents": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["raw_documents"])
        docs = data["documents"]
        
        rows = []
        for doc in docs[:10]:  # Show first 10 documents
            title = doc["title"][:50] + "..." if len(doc["title"]) > 50 else doc["title"]
            content_preview = doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"]
            rows.append([doc["id"], title, len(doc["content"]), content_preview])
        
        return {
            "columns": ["ID", "Title", "Length", "Content Preview"],
            "rows": rows,
            "total_rows": data["total_documents"],
            "total_characters": data["total_characters"],
            "avg_length": round(data["average_length"], 1)
        }
```

## Card 2: Text Chunker

**File:** `text_chunker.py` | **Folder:** `data/`

Splits documents into smaller chunks for better retrieval and processing.

```python
from cards.base import BaseCard
import re
from typing import List

class TextChunkerCard(BaseCard):
    card_type = "rag_text_chunker"
    display_name = "Text Chunker"
    description = "Split documents into chunks"
    category = "data" 
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "chunk_size": {
            "type": "number",
            "label": "Chunk size (characters)",
            "default": 500
        },
        "overlap_size": {
            "type": "number",
            "label": "Overlap between chunks (characters)",
            "default": 50
        },
        "chunking_strategy": {
            "type": "string", 
            "label": "Chunking strategy (fixed, sentence, paragraph)",
            "default": "sentence"
        }
    }
    input_schema = {"raw_documents": "json"}
    output_schema = {"text_chunks": "json"}

    def execute(self, config, inputs, storage):
        doc_data = storage.load_json(inputs["raw_documents"])
        documents = doc_data["documents"]
        
        chunk_size = int(config.get("chunk_size", 500))
        overlap_size = int(config.get("overlap_size", 50))
        strategy = config.get("chunking_strategy", "sentence")
        
        all_chunks = []
        
        for doc in documents:
            content = doc["content"]
            doc_id = doc["id"]
            
            if strategy == "fixed":
                chunks = self._chunk_fixed_size(content, chunk_size, overlap_size)
            elif strategy == "sentence":
                chunks = self._chunk_by_sentences(content, chunk_size, overlap_size)
            elif strategy == "paragraph":
                chunks = self._chunk_by_paragraphs(content, chunk_size, overlap_size)
            else:
                chunks = self._chunk_fixed_size(content, chunk_size, overlap_size)
            
            # Add metadata to chunks
            for i, chunk_text in enumerate(chunks):
                chunk = {
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "document_id": doc_id,
                    "document_title": doc["title"],
                    "chunk_index": i,
                    "content": chunk_text,
                    "character_count": len(chunk_text),
                    "source_metadata": doc.get("metadata", {}),
                    "source": doc["source"]
                }
                all_chunks.append(chunk)
        
        # Calculate statistics
        chunk_lengths = [len(chunk["content"]) for chunk in all_chunks]
        
        chunk_data = {
            "chunks": all_chunks,
            "total_chunks": len(all_chunks),
            "original_documents": len(documents),
            "chunk_stats": {
                "min_length": min(chunk_lengths) if chunk_lengths else 0,
                "max_length": max(chunk_lengths) if chunk_lengths else 0,
                "avg_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                "total_characters": sum(chunk_lengths)
            },
            "chunking_config": {
                "chunk_size": chunk_size,
                "overlap_size": overlap_size,
                "strategy": strategy
            }
        }
        
        ref = storage.save_json("_p", "_n", "text_chunks", chunk_data)
        return {"text_chunks": ref}

    def _chunk_fixed_size(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks

    def _chunk_by_sentences(self, text: str, target_size: int, overlap: int) -> List[str]:
        """Split text by sentences, grouping to approximate target size."""
        # Simple sentence splitting (in production, use spaCy or NLTK)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed target size, save current chunk
            if len(current_chunk) + len(sentence) > target_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _chunk_by_paragraphs(self, text: str, target_size: int, overlap: int) -> List[str]:
        """Split text by paragraphs, grouping to approximate target size."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > target_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["text_chunks"])
        chunks = data["chunks"][:10]
        stats = data["chunk_stats"]
        
        rows = []
        for chunk in chunks:
            content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
            rows.append([
                chunk["chunk_id"],
                chunk["document_title"][:30] + "..." if len(chunk["document_title"]) > 30 else chunk["document_title"],
                chunk["character_count"],
                content_preview
            ])
        
        return {
            "columns": ["Chunk ID", "Document", "Length", "Content Preview"],
            "rows": rows,
            "total_rows": data["total_chunks"], 
            "avg_chunk_length": round(stats["avg_length"], 1),
            "min_length": stats["min_length"],
            "max_length": stats["max_length"]
        }
```

## Card 3: Vector Embedder

**File:** `vector_embedder.py` | **Folder:** `embedding/`

Converts text chunks into vector embeddings using a sentence transformer model.

```python
from cards.base import BaseCard
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class VectorEmbedderCard(BaseCard):
    card_type = "rag_vector_embedder"
    display_name = "Vector Embedder"
    description = "Convert text chunks to embeddings"
    category = "embedding"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "model_name": {
            "type": "string",
            "label": "Sentence transformer model",
            "default": "all-MiniLM-L6-v2"
        },
        "batch_size": {
            "type": "number",
            "label": "Embedding batch size",
            "default": 32
        },
        "normalize": {
            "type": "boolean",
            "label": "Normalize embeddings (for cosine similarity)",
            "default": True
        }
    }
    input_schema = {"text_chunks": "json"}
    output_schema = {"embeddings": "json"}

    def execute(self, config, inputs, storage):
        chunk_data = storage.load_json(inputs["text_chunks"])
        chunks = chunk_data["chunks"]
        
        model_name = config.get("model_name", "all-MiniLM-L6-v2")
        batch_size = int(config.get("batch_size", 32))
        normalize = config.get("normalize", True)
        
        # Load sentence transformer model
        model = SentenceTransformer(model_name)
        
        # Extract text content
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings in batches
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch_texts,
                normalize_embeddings=normalize,
                show_progress_bar=True if i == 0 else False
            )
            all_embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        
        # Create embedding data with metadata
        embedding_records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            record = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "content": chunk["content"],
                "embedding": embedding.tolist(),
                "embedding_index": i,
                "source_metadata": chunk["source_metadata"],
                "character_count": chunk["character_count"]
            }
            embedding_records.append(record)
        
        # Calculate embedding statistics
        embedding_stats = {
            "num_embeddings": len(embeddings_array),
            "embedding_dim": embeddings_array.shape[1],
            "model_used": model_name,
            "normalized": normalize,
            "mean_norm": float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings_array, axis=1)))
        }
        
        # Sample similarity analysis
        if len(embeddings_array) >= 2:
            # Compute pairwise similarities for first few embeddings
            sample_size = min(5, len(embeddings_array))
            sample_embeddings = embeddings_array[:sample_size]
            similarities = np.dot(sample_embeddings, sample_embeddings.T)
            
            embedding_stats["sample_similarities"] = {
                "min_similarity": float(np.min(similarities[similarities < 0.99])),  # Exclude self-similarity
                "max_similarity": float(np.max(similarities[similarities < 0.99])),
                "avg_similarity": float(np.mean(similarities[similarities < 0.99]))
            }
        
        embedding_data = {
            "embeddings": embedding_records,
            "embedding_stats": embedding_stats,
            "model_config": {
                "model_name": model_name,
                "batch_size": batch_size,
                "normalize": normalize
            },
            "original_chunks": len(chunks)
        }
        
        ref = storage.save_json("_p", "_n", "embeddings", embedding_data)
        return {"embeddings": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["embeddings"])
        stats = data["embedding_stats"]
        embeddings = data["embeddings"][:5]
        
        # Show embedding previews
        rows = []
        for emb in embeddings:
            content_preview = emb["content"][:100] + "..." if len(emb["content"]) > 100 else emb["content"]
            embedding_preview = str(emb["embedding"][:3])[:-1] + ", ...]"
            rows.append([
                emb["chunk_id"],
                content_preview,
                len(emb["embedding"]),
                embedding_preview
            ])
        
        return {
            "columns": ["Chunk ID", "Content Preview", "Embedding Dim", "Embedding Preview"],
            "rows": rows,
            "total_rows": stats["num_embeddings"],
            "embedding_dimension": stats["embedding_dim"],
            "model_used": stats["model_used"],
            "normalized": "Yes" if stats["normalized"] else "No",
            "mean_norm": round(stats["mean_norm"], 4)
        }
```

## Card 4: Vector Store

**File:** `vector_store.py` | **Folder:** `embedding/`

Creates a searchable vector database from the embeddings.

```python
from cards.base import BaseCard
import numpy as np
import faiss
from typing import List, Tuple
import json

class VectorStoreCard(BaseCard):
    card_type = "rag_vector_store"
    display_name = "Vector Store"
    description = "Create searchable vector database"
    category = "embedding"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "index_type": {
            "type": "string",
            "label": "FAISS index type (flat, ivf)",
            "default": "flat"
        },
        "metric": {
            "type": "string", 
            "label": "Distance metric (cosine, euclidean)",
            "default": "cosine"
        },
        "nlist": {
            "type": "number",
            "label": "Number of clusters (for IVF index)",
            "default": 100
        }
    }
    input_schema = {"embeddings": "json"}
    output_schema = {"vector_db": "json"}

    def execute(self, config, inputs, storage):
        embedding_data = storage.load_json(inputs["embeddings"])
        embeddings = embedding_data["embeddings"]
        
        index_type = config.get("index_type", "flat")
        metric = config.get("metric", "cosine")
        nlist = int(config.get("nlist", 100))
        
        # Extract embedding vectors and metadata
        vectors = np.array([emb["embedding"] for emb in embeddings]).astype('float32')
        dimension = vectors.shape[1]
        
        # Create FAISS index
        if metric == "cosine":
            # For cosine similarity, we can use inner product if vectors are normalized
            if index_type == "flat":
                index = faiss.IndexFlatIP(dimension)  # Inner Product
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, min(nlist, len(vectors)))
                index.train(vectors)
            else:
                index = faiss.IndexFlatIP(dimension)
        else:  # euclidean
            if index_type == "flat":
                index = faiss.IndexFlatL2(dimension)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, min(nlist, len(vectors)))
                index.train(vectors)
            else:
                index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to index
        index.add(vectors)
        
        # Create metadata lookup
        metadata_lookup = {}
        for i, emb in enumerate(embeddings):
            metadata_lookup[i] = {
                "chunk_id": emb["chunk_id"],
                "document_id": emb["document_id"],
                "document_title": emb["document_title"],
                "content": emb["content"],
                "character_count": emb["character_count"],
                "source_metadata": emb["source_metadata"]
            }
        
        # Test search functionality
        if len(vectors) > 0:
            # Search with first vector as query
            test_query = vectors[0:1]
            distances, indices = index.search(test_query, min(5, len(vectors)))
            
            test_results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # Valid result
                    test_results.append({
                        "rank": i + 1,
                        "distance": float(dist),
                        "chunk_id": metadata_lookup[idx]["chunk_id"],
                        "content_preview": metadata_lookup[idx]["content"][:100] + "..." 
                    })
        else:
            test_results = []
        
        # Serialize index for storage (simplified approach)
        # In production, you'd save the FAISS index to a file
        vector_db_data = {
            "index_info": {
                "index_type": index_type,
                "metric": metric,
                "dimension": dimension,
                "total_vectors": index.ntotal,
                "is_trained": index.is_trained
            },
            "metadata_lookup": metadata_lookup,
            "config": {
                "index_type": index_type,
                "metric": metric,
                "nlist": nlist
            },
            "vectors": vectors.tolist(),  # Store vectors for rebuilding index
            "test_search": test_results,
            "embedding_stats": embedding_data["embedding_stats"]
        }
        
        ref = storage.save_json("_p", "_n", "vector_db", vector_db_data)
        return {"vector_db": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["vector_db"])
        index_info = data["index_info"]
        test_results = data["test_search"]
        
        # Show test search results
        search_rows = []
        for result in test_results[:5]:
            search_rows.append([
                result["rank"],
                result["chunk_id"],
                round(result["distance"], 4),
                result["content_preview"]
            ])
        
        return {
            "index_type": index_info["index_type"],
            "metric": data["config"]["metric"],
            "total_vectors": index_info["total_vectors"],
            "dimension": index_info["dimension"],
            "is_trained": "Yes" if index_info["is_trained"] else "No",
            "test_search_results": {
                "columns": ["Rank", "Chunk ID", "Distance", "Content Preview"],
                "rows": search_rows
            }
        }
```

## Card 5: RAG Query

**File:** `rag_query.py` | **Folder:** `retrieval/`

Performs similarity search to retrieve relevant document chunks for a query.

```python
from cards.base import BaseCard
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class RAGQueryCard(BaseCard):
    card_type = "rag_query"
    display_name = "RAG Query"
    description = "Retrieve relevant documents for query"
    category = "retrieval"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "query": {
            "type": "string",
            "label": "Search query",
            "default": "What is machine learning and how does it work?"
        },
        "top_k": {
            "type": "number",
            "label": "Number of results to retrieve",
            "default": 5
        },
        "min_similarity": {
            "type": "number",
            "label": "Minimum similarity threshold",
            "default": 0.1
        }
    }
    input_schema = {"vector_db": "json"}
    output_schema = {"retrieved_context": "json"}

    def execute(self, config, inputs, storage):
        vector_db_data = storage.load_json(inputs["vector_db"])
        
        query = config.get("query", "What is machine learning and how does it work?")
        top_k = int(config.get("top_k", 5))
        min_similarity = float(config.get("min_similarity", 0.1))
        
        # Rebuild the FAISS index from stored data
        vectors = np.array(vector_db_data["vectors"]).astype('float32')
        dimension = vector_db_data["index_info"]["dimension"]
        metric = vector_db_data["config"]["metric"]
        index_type = vector_db_data["config"]["index_type"]
        
        # Recreate index
        if metric == "cosine":
            if index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            else:
                quantizer = faiss.IndexFlatIP(dimension)
                nlist = vector_db_data["config"]["nlist"]
                index = faiss.IndexIVFFlat(quantizer, dimension, min(nlist, len(vectors)))
                index.train(vectors)
        else:  # euclidean
            if index_type == "flat":
                index = faiss.IndexFlatL2(dimension)
            else:
                quantizer = faiss.IndexFlatL2(dimension)
                nlist = vector_db_data["config"]["nlist"]
                index = faiss.IndexIVFFlat(quantizer, dimension, min(nlist, len(vectors)))
                index.train(vectors)
        
        index.add(vectors)
        
        # Load embedding model (same as used in vector embedder)
        model_name = vector_db_data["embedding_stats"]["model_used"]
        model = SentenceTransformer(model_name)
        
        # Generate query embedding
        query_embedding = model.encode(
            [query], 
            normalize_embeddings=vector_db_data["config"]["metric"] == "cosine"
        )
        query_vector = query_embedding.astype('float32')
        
        # Perform search
        distances, indices = index.search(query_vector, top_k)
        
        # Retrieve results with metadata
        metadata_lookup = vector_db_data["metadata_lookup"]
        retrieved_chunks = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid result
                # Convert distance to similarity score
                if metric == "cosine":
                    similarity = float(distance)  # Inner product
                else:
                    # Convert L2 distance to similarity (higher = more similar)
                    similarity = 1 / (1 + float(distance))
                
                if similarity >= min_similarity:
                    chunk_metadata = metadata_lookup[str(idx)]
                    
                    retrieved_chunks.append({
                        "rank": i + 1,
                        "similarity_score": similarity,
                        "chunk_id": chunk_metadata["chunk_id"],
                        "document_id": chunk_metadata["document_id"],
                        "document_title": chunk_metadata["document_title"],
                        "content": chunk_metadata["content"],
                        "character_count": chunk_metadata["character_count"],
                        "source_metadata": chunk_metadata["source_metadata"]
                    })
        
        # Prepare context for generation
        context_text = "\n\n".join([chunk["content"] for chunk in retrieved_chunks])
        
        # Statistics
        retrieval_stats = {
            "query": query,
            "total_retrieved": len(retrieved_chunks),
            "top_k_requested": top_k,
            "min_similarity": min_similarity,
            "context_length": len(context_text),
            "unique_documents": len(set(chunk["document_id"] for chunk in retrieved_chunks)),
            "avg_similarity": np.mean([chunk["similarity_score"] for chunk in retrieved_chunks]) if retrieved_chunks else 0
        }
        
        retrieval_data = {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "context_text": context_text,
            "retrieval_stats": retrieval_stats,
            "search_config": {
                "top_k": top_k,
                "min_similarity": min_similarity,
                "metric": metric
            }
        }
        
        ref = storage.save_json("_p", "_n", "retrieved_context", retrieval_data)
        return {"retrieved_context": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["retrieved_context"])
        chunks = data["retrieved_chunks"][:10]
        stats = data["retrieval_stats"]
        
        rows = []
        for chunk in chunks:
            content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
            rows.append([
                chunk["rank"],
                chunk["document_title"][:30] + "..." if len(chunk["document_title"]) > 30 else chunk["document_title"],
                round(chunk["similarity_score"], 4),
                content_preview
            ])
        
        return {
            "columns": ["Rank", "Document", "Similarity", "Content Preview"],
            "rows": rows,
            "total_rows": stats["total_retrieved"],
            "query": stats["query"][:100] + "..." if len(stats["query"]) > 100 else stats["query"],
            "avg_similarity": round(stats["avg_similarity"], 4),
            "context_length": stats["context_length"],
            "unique_documents": stats["unique_documents"]
        }
```

## Card 6: Response Generator

**File:** `response_generator.py` | **Folder:** `generation/`

Generates final response using retrieved context and a language model.

```python
from cards.base import BaseCard
import openai
import os

class ResponseGeneratorCard(BaseCard):
    card_type = "rag_response_generator"
    display_name = "Response Generator"
    description = "Generate response using retrieved context"
    category = "generation"
    execution_mode = "local"
    output_view_type = "text"

    config_schema = {
        "model_type": {
            "type": "string",
            "label": "Model type (openai, huggingface, local)",
            "default": "local"
        },
        "openai_model": {
            "type": "string",
            "label": "OpenAI model (if using OpenAI)",
            "default": "gpt-3.5-turbo"
        },
        "max_tokens": {
            "type": "number",
            "label": "Max response tokens",
            "default": 500
        },
        "temperature": {
            "type": "number",
            "label": "Temperature (creativity)",
            "default": 0.7
        },
        "custom_prompt": {
            "type": "string",
            "label": "Custom system prompt (optional)",
            "default": ""
        }
    }
    input_schema = {"retrieved_context": "json"}
    output_schema = {"rag_response": "json"}

    def execute(self, config, inputs, storage):
        context_data = storage.load_json(inputs["retrieved_context"])
        
        query = context_data["query"]
        context_text = context_data["context_text"] 
        retrieved_chunks = context_data["retrieved_chunks"]
        
        model_type = config.get("model_type", "local")
        max_tokens = int(config.get("max_tokens", 500))
        temperature = float(config.get("temperature", 0.7))
        custom_prompt = config.get("custom_prompt", "")
        
        # Build prompt
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Use only the information from the context to answer the question. If the context doesn't contain 
            enough information to answer the question, say so. Provide accurate and concise responses."""
        
        user_prompt = f"""Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""

        if model_type == "openai":
            # Use OpenAI API
            try:
                openai_model = config.get("openai_model", "gpt-3.5-turbo")
                
                # This would require an API key
                # response = openai.ChatCompletion.create(
                #     model=openai_model,
                #     messages=[
                #         {"role": "system", "content": system_prompt},
                #         {"role": "user", "content": user_prompt}
                #     ],
                #     max_tokens=max_tokens,
                #     temperature=temperature
                # )
                # generated_response = response.choices[0].message.content
                
                # For demo, return a placeholder
                generated_response = f"""[OpenAI API Response - Would use {openai_model}]

                Based on the retrieved context about machine learning and AI, I can provide the following information:

                {context_text[:500]}...

                This response would be generated using the OpenAI API with the retrieved context."""
                
            except Exception as e:
                generated_response = f"Error calling OpenAI API: {str(e)}"
                
        elif model_type == "huggingface":
            # Use HuggingFace transformers (would require model loading)
            generated_response = f"""[HuggingFace Model Response]
            
            Based on the retrieved documents, here's what I found about your question: "{query}"
            
            {context_text[:800]}...
            
            This demonstrates how a local HuggingFace model would generate a response using the retrieved context."""
            
        else:  # local/simple
            # Simple template-based response for demonstration
            if not context_text.strip():
                generated_response = "I couldn't find any relevant information to answer your question. Please try rephrasing your query or ensure that relevant documents are available in the knowledge base."
            else:
                # Extract key information and create response
                sentences = context_text.split('. ')
                key_info = '. '.join(sentences[:3])  # First 3 sentences
                
                generated_response = f"""Based on the available information, I can answer your question: "{query}"

{key_info}

The retrieved documents provide additional context about this topic. Would you like me to elaborate on any specific aspect?

Sources: {len(retrieved_chunks)} relevant document chunks were used to generate this response."""

        # Count tokens (simple approximation)
        response_tokens = len(generated_response.split())
        context_tokens = len(context_text.split())
        
        # Analyze response quality
        response_stats = {
            "query": query,
            "response_length": len(generated_response),
            "response_tokens": response_tokens,
            "context_tokens": context_tokens,
            "chunks_used": len(retrieved_chunks),
            "model_type": model_type,
            "temperature": temperature,
            "contains_context_info": context_text.lower() in generated_response.lower() if context_text else False
        }
        
        response_data = {
            "query": query,
            "response": generated_response,
            "retrieved_chunks": retrieved_chunks,
            "context_text": context_text,
            "response_stats": response_stats,
            "generation_config": {
                "model_type": model_type,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "custom_prompt": custom_prompt
            }
        }
        
        ref = storage.save_json("_p", "_n", "rag_response", response_data)
        return {"rag_response": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["rag_response"])
        stats = data["response_stats"]
        
        # Show response preview
        response_preview = data["response"][:500] + "..." if len(data["response"]) > 500 else data["response"]
        
        return {
            "query": data["query"],
            "response_preview": response_preview,
            "response_length": stats["response_length"],
            "response_tokens": stats["response_tokens"],
            "chunks_used": stats["chunks_used"],
            "model_type": stats["model_type"],
            "context_tokens": stats["context_tokens"]
        }
```

---

## How to Wire the Pipeline

### Canvas Connections:

```
[Document Loader] ──> [Text Chunker] ──> [Vector Embedder] ──> [Vector Store]
                                                                      │
[RAG Query] <─────────────────────────────────────────────────────────┘
     │
     │
     └──> [Response Generator]
```

### Key Configuration:

- **Document Loader**: Choose sample data, URLs, or direct text input
- **Text Chunker**: Set chunk size (300-1000 chars) and overlap (50-100 chars)
- **Vector Embedder**: Select sentence transformer model (all-MiniLM-L6-v2 recommended)
- **Vector Store**: Choose FAISS index type (flat for small datasets, IVF for large)
- **RAG Query**: Enter your question and set top_k retrieval count
- **Response Generator**: Choose model type and configure generation parameters

### Sample Queries to Try:

1. "What is machine learning and how does it work?"
2. "Explain the different types of machine learning"
3. "What are the ethical considerations in AI development?"
4. "How do neural networks and deep learning relate?"
5. "What are the challenges with bias in AI systems?"

This pipeline provides a complete RAG system for question-answering over document collections using semantic search and language generation.