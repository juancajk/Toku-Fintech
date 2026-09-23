# 🏛️ Asistente RAG Normativo Ley Fintech - Toku

Sistema inteligente basado en Modelos de Lenguaje (LLM) y recuperación aumentada (RAG) 100% local, diseñado para automatizar y blindar jurídicamente la interpretación de normativas y la Ley N° 21.521 (Ley Fintech) para la startup chilena **Toku**.

## Contexto Organizacional y Desafío (Caso de Negocio)
- **Organización**: Toku (Startup chilena del sector Fintech enfocada en la optimización de pagos recurrentes y soluciones B2B).
- **Desafío Regulatorio**: El ecosistema Fintech nacional enfrenta el reto de "crecer bajo regulación" y gobernar los riesgos operativos derivados de la implementación del Sistema de Finanzas Abiertas (SFA) exigido por la CMF y la Ley Fintech. Las áreas legales y técnicas manejan un volumen inmanejable de normativas donde un error implica multas severas.
- **Solución RAG**: Un agente autónomo que elimina las alucinaciones normativas, entregando respuestas inmediatas fundamentadas exclusivamente en los textos oficiales de la CMF con **trazabilidad obligatoria** (archivo de origen y número de página).

## Estructura del Proyecto
```text
├── documentos/            # Directorio de normativas oficiales en formato PDF (Ley Fintech)
├── vector_db/             # Base de datos vectorial persistente (ChromaDB)
├── src/
│   └── main.py            # Lógica principal del pipeline RAG, chunking y prompt estricto
├── requirements.txt       # Dependencias y librerías del proyecto
└── README.md              # Documentación técnica de ejecución

