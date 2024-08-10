from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = PyPDFLoader("data.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap= 50
)
texts = text_splitter.split_documents(documents)
model_name = "BAAI/bge-large-en"
model_kwargs = {
    'device': 'cpu'
}
encode_kwargs = {
    'normalize_embeddings': False
}
embeddings = HuggingFaceBgeEmbeddings(
    model_kwargs=model_kwargs,
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

url="https://309e7711-786b-4c2c-aa32-6fff465f6bf9.us-east4-0.gcp.cloud.qdrant.io:6333",
api_key="481cCVm0Ks14GPiWtcjv9zWFsNreetkCjdBcUg3X19_IaxiG80756w",
collection_name = "Gyber_db"

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name=collection_name
)