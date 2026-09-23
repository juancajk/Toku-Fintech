import os
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

def construir_pipeline_rag():
    pdf_path = "./documentos/Ley_Fintech_y_Normas_CMF.pdf"

    if not os.path.exists(pdf_path):
        print("Error: No se encuentra el archivo PDF.")
        return None

    print("[1/4] Cargando documentos normativos oficiales de Toku...")
    loader = PyPDFLoader(pdf_path)
    documentos = loader.load()

    print("[2/4] Aplicando estrategia de Chunking...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(documentos)

    print("[3/4] Generando embeddings locales (ChromaDB)...")
    embeddings_locales = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_locales,
        persist_directory="./vector_db",
    )

    retriever = vector_db.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    print("[4/4] Cargando modelo de lenguaje 100% local en tu PC...")
    template = """Eres un asistente legal especializado en la Ley N° 21.521 para Toku.

DIRECTRICES ESTRICTAS:
1. Utiliza EXCLUSIVAMENTE el contexto normativo recuperado para responder.
2. Esta terminantemente prohibido inventar informacion.
3. Si la respuesta no esta en el contexto, indica: "La informacion no se encuentra disponible."

Contexto recuperado:
{context}

Consulta del usuario:
{question}

Respuesta fundamentada:"""

    CUSTOM_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 300, "temperature": 0.1}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    )

    return qa_chain

if __name__ == "__main__":
    print("=========================================================")
    print("AGENTE RAG TOKU - SISTEMA 100% LOCAL Y ROBUSTO")
    print("=========================================================")

    agente = construir_pipeline_rag()

    if agente:
        consulta = "¿Cuales son los requisitos exigidos por la ley para la prestacion de servicios financieros basados en tecnologia?"
        print(f"Consulta de prueba: '{consulta}'\n")

        try:
            resultado = agente.invoke({"query": consulta})
            
            print("-------------------- RESPUESTA DEL AGENTE --------------------")
            print(resultado["result"])
            print("--------------------------------------------------------------")

            print("EVIDENCIA DE TRAZABILIDAD (Fuentes de origen):")
            for idx, doc in enumerate(resultado["source_documents"], 1):
                print(
                    f"[{idx}] Archivo: {doc.metadata.get('source', 'N/A')} | Pagina: {doc.metadata.get('page', 'N/A')}"
                )
            print("=========================================================")
        except Exception as e:
            print(f"ERROR: {e}")