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
        print("Error: No se encuentra el archivo PDF en la ruta especificada.")
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
    
    
    template = """Eres un asistente legal estrictamente limitado al contexto provisto.

REGLAS ABSOLUTAS:
1. Si la respuesta exacta no se encuentra explícitamente en el "Contexto recuperado", debes responder ÚNICAMENTE: "La información no se encuentra disponible."
2. Está terminantemente prohibido inventar cifras, artículos, montos o explicaciones.

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
        pipeline_kwargs={
            "max_new_tokens": 150, 
            "temperature": 0.0, 
            "do_sample": False
        }
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
    print("AGENTE RAG TOKU - DEMOSTRACIÓN OFICIAL ISY0101")
    print("=========================================================")

    agente = construir_pipeline_rag()

    if agente:
        consultas_prueba = [
            "¿Cuáles son los requisitos exigidos por la ley para la prestación de servicios financieros basados en tecnología?",
            "¿Cuántos impuestos en criptomonedas debe pagar Toku según el artículo 99?" # Prueba anti-alucinación
        ]

        for idx, consulta in enumerate(consultas_prueba, 1):
            print(f"\n[PRUEBA {idx}/2] Consulta: '{consulta}'\n")
            try:
                resultado = agente.invoke({"query": consulta})
                
                print("-------------------- RESPUESTA DEL AGENTE --------------------")
                print(resultado["result"])
                print("--------------------------------------------------------------")

                print("TRAZABILIDAD INSTITUCIONAL (Fuentes de origen):")
                for f_idx, doc in enumerate(resultado["source_documents"], 1):
                    print(
                        f"  [{f_idx}] Archivo: {doc.metadata.get('source', 'N/A')} | Página: {doc.metadata.get('page', 'N/A')}"
                    )
            except Exception as e:
                print(f"ERROR: {e}")
            print("=" * 65)
            