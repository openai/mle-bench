"""
Embedding Models Module

Provides unified interface for different embedding model types:
- Local: sentence-transformers models
- OpenAI: OpenAI embedding API
- Azure: Azure OpenAI embedding API
- Custom: User-defined embedding services
"""

# Disable TensorFlow backend before any imports
import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import numpy as np
from typing import List, Literal, Optional


class EmbeddingModel:
    """
    Unified embedding model supporting both local and remote APIs

    Supported types:
    - local: sentence-transformers models
    - openai: OpenAI embedding API
    - azure: Azure OpenAI embedding API
    - custom: Custom API endpoint
    """

    def __init__(
        self,
        model_type: Literal["local", "openai", "azure", "custom"] = "local",
        model_name: str = "gte-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize embedding model

        Args:
            model_type: Type of model (local/openai/azure/custom)
            model_name: Model name or identifier
            api_key: API key for remote models
            base_url: Base URL for API endpoints
            dimension: Embedding dimension (auto-detected for local models)
            device: Device to use for local models ("cpu" or "cuda")
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.device = device or "cpu"
        self.kwargs = kwargs

        if model_type == "local":
            from sentence_transformers import SentenceTransformer
            model_path = '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5965-xiaokangcheng/models/gte-small'
            self.model = SentenceTransformer(model_path)
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
        elif model_type == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            # OpenAI text-embedding-3-small: 1536, text-embedding-ada-002: 1536
            self.dimension = dimension or 1536
        elif model_type == "azure":
            import openai
            self.client = openai.AzureOpenAI(
                api_key=api_key,
                api_version=kwargs.get("api_version", "2023-05-15"),
                azure_endpoint=base_url
            )
            self.dimension = dimension or 1536
        elif model_type == "custom":
            # Custom API - user needs to implement their own logic
            self.dimension = dimension
            if not dimension:
                raise ValueError("dimension must be specified for custom model type")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings
            show_progress_bar: Show progress bar (only for local models)

        Returns:
            embeddings: numpy array of shape (len(texts), dimension)
        """
        if self.model_type == "local":
            embeddings = self.model.encode(texts, show_progress_bar=show_progress_bar)
            return np.array(embeddings, dtype=np.float32)

        elif self.model_type == "openai":
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)

        elif self.model_type == "azure":
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)

        elif self.model_type == "custom":
            # Placeholder for custom implementation
            raise NotImplementedError(
                "Custom embedding model requires user implementation. "
                "Please subclass EmbeddingModel and override the encode method."
            )

        return np.array([])
