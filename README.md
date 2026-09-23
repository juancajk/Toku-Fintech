# Asistente RAG Normativo Ley Fintech - Toku

Sistema inteligente basado en Modelos de Lenguaje (LLM) y recuperación aumentada (RAG) 100% local, diseñado para automatizar y blindar jurídicamente la interpretación de normativas y la Ley N° 21.521 (Ley Fintech) para la startup chilena **Toku**.

## Estructura del Proyecto
```text
├── documentos/            # Directorio de normativas oficiales en formato PDF (Ley Fintech)
├── vector_db/             # Base de datos vectorial persistente (ChromaDB)
├── src/
│   └── main.py            # Lógica principal del pipeline RAG, chunking y prompt estricto
├── requirements.txt       # Dependencias y librerías del proyecto
└── README.md              # Documentación técnica de ejecución