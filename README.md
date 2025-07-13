# Multimodal RAG - Sistema di Retrieval Augmented Generation Multimodale

Un sistema avanzato di RAG (Retrieval Augmented Generation) che combina elaborazione di testo e immagini per analizzare documenti PDF tecnici e scientifici.

## 🚀 Caratteristiche Principali

### **Elaborazione Multimodale**

- **Testo**: Estrazione e chunking intelligente del testo dai PDF
- **Immagini**: Estrazione automatica di immagini ad alta risoluzione (300 DPI)
- **Tabelle**: Parsing automatico delle tabelle HTML con conversione in formato markdown
- **Metadati Ricchi**: Tracciamento completo delle fonti con informazioni su pagina, sezione e tipo di contenuto

### **Modelli AI Integrati**

- **Embeddings Testo**: BGE-base-en-v1.5 per ricerca semantica del testo
- **Embeddings Immagini**: CLIP ViT-B-32 per ricerca visiva
- **Captioning**: Gemini 2.5 Flash per descrizioni dettagliate delle immagini
- **LLM**: Gemini 2.5 Flash per generazione di risposte contestuali

### **Architettura Ibrida**

- **Retriever Ibrido**: Combina ricerca testuale e visiva
- **Vector Database**: Chroma per storage persistente degli embeddings
- **Deduplicazione**: Eliminazione automatica di documenti duplicati
- **Citing**: Citazioni precise delle fonti con metadati completi

## 📋 Prerequisiti

### **Sistema Operativo**

- Linux (testato su Ubuntu 24.04 WSL2)
- Windows con WSL2
- macOS

### **Hardware Consigliato**

- **RAM**: Minimo 8GB, consigliato 16GB+
- **GPU**: Opzionale ma consigliata per CLIP (CUDA supportato)
- **Storage**: Spazio sufficiente per gli indici Chroma e le immagini estratte

### **Software**

- Python 3.8+
- pip (gestore pacchetti Python)

## 🛠️ Installazione

### 1. **Clona il Repository**

```bash
git clone <repository-url>
cd ai_model_ISO
```

### 2. **Crea un Ambiente Virtuale**

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# oppure
venv\Scripts\activate     # Windows
```

### 3. **Installa le Dipendenze**

```bash
pip install -r requirements.txt
```

### 4. **Configura le API Keys**

Crea un file `.env` nella root del progetto:

```bash
# Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here
```

**Nota**: Ottieni la tua API key da [Google AI Studio](https://makersuite.google.com/app/apikey)

## 📖 Utilizzo

### **Struttura delle Cartelle**

```
ai_model_ISO/
├── train_model.py          # Script principale
├── requirements.txt        # Dipendenze Python
├── .env                   # Configurazione API keys
├── pdfs/                  # Cartella con i PDF da analizzare
│   ├── documento1.pdf
│   ├── documento2.pdf
│   └── ...
└── iso_index/             # Indici Chroma (generati automaticamente)
    ├── _txt/              # Indice per il testo
    └── _img/              # Indice per le immagini
```

### **Esecuzione del Sistema**

#### **Comando Base**

```bash
python train_model.py --pdf_dir ./pdfs
```

#### **Parametri Avanzati**

```bash
python train_model.py \
    --pdf_dir ./pdfs \
    --persist_dir ./custom_index \
    --k 20
```

#### **Parametri Disponibili**

- `--pdf_dir`: Cartella contenente i PDF da processare (obbligatorio)
- `--persist_dir`: Directory per salvare gli indici Chroma (default: `./iso_index`)
- `--k`: Numero di documenti da recuperare per query (default: 15)

### **Workflow di Elaborazione**

1. **Estrazione PDF**: Il sistema processa ricorsivamente tutti i PDF nella cartella specificata
2. **Chunking Testo**: Il testo viene diviso in chunk di 800 caratteri con overlap di 100
3. **Estrazione Immagini**: Ogni pagina viene convertita in immagine PNG a 300 DPI
4. **Captioning**: Gemini genera descrizioni dettagliate di ogni immagine
5. **Embeddings**: Vengono creati embeddings per testo e immagini
6. **Indexing**: Gli embeddings vengono salvati in Chroma DB
7. **RAG Ready**: Il sistema è pronto per rispondere alle domande

### **Interfaccia Interattiva**

Dopo l'elaborazione, il sistema avvia un'interfaccia interattiva:

```
Multimodal RAG pronto → premi Invio vuoto per uscire

Domanda › Quali sono i requisiti di sicurezza per i parastinchi?
```

#### **Esempio di Risposta**

```
▸ Risposta:
Secondo gli standard tecnici analizzati, i parastinchi devono soddisfare i seguenti requisiti di sicurezza:

1. **Resistenza agli urti**: Capacità di assorbire impatti fino a 50 J
2. **Materiali**: Composizione in polietilene ad alta densità (HDPE)
3. **Dimensioni**: Copertura minima del 70% della superficie tibiale
4. **Fissaggio**: Sistema di ancoraggio che previene lo spostamento durante l'uso

▸ Fonti:
  • I parastinchi devono essere realizzati in materiali... (Source: standard_ISO_12345.pdf, p.15, Section: Requisiti Materiali)
  • Le dimensioni minime sono definite... (Source: standard_ISO_12345.pdf, p.18, Type: Table)
  • Immagine: Diagramma delle zone di protezione... (Source: standard_ISO_12345.pdf, p.22, Type: Image Caption)
```

## 🔧 Configurazione Avanzata

### **Modifica dei Prompt**

I prompt sono configurabili nel codice:

- **Prompt Captioning**: `caption_image_with_gemini()` - Personalizza la descrizione delle immagini
- **Prompt QA**: `qa_prompt` - Personalizza il comportamento del LLM
- **Document Prompt**: `document_prompt` - Personalizza il formato delle citazioni

### **Parametri di Chunking**

Modifica i parametri di chunking in `load_pdf()`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Dimensione chunk
    chunk_overlap=100    # Overlap tra chunk
)
```

### **Modelli di Embedding**

Cambia i modelli di embedding:

```python
# Per il testo
txt_embed = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",  # Cambia modello
    encode_kwargs={"normalize_embeddings": True}
)

# Per le immagini (in CLIPEmbeddings)
self.model, _, self.pre = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"  # Cambia modello CLIP
)
```

## 📊 Performance e Ottimizzazioni

### **GPU Acceleration**

Il sistema supporta automaticamente CUDA se disponibile:

- CLIP embeddings: Accelerazione GPU automatica
- PyTorch: Utilizzo GPU per operazioni tensoriali

### **Memoria e Storage**

- **Indici Chroma**: ~100MB per 1000 documenti
- **Immagini estratte**: ~5-10MB per PDF (dipende dal numero di pagine)
- **RAM**: Picco durante l'elaborazione ~4-8GB

### **Tempi di Elaborazione**

- **PDF Processing**: ~30-60 secondi per PDF (dipende dal numero di pagine)
- **Captioning**: ~5-10 secondi per immagine (dipende dalla complessità)
- **Indexing**: ~10-30 secondi per 1000 documenti

## 🐛 Risoluzione Problemi

### **Errori Comuni**

#### **API Key Non Configurata**

```
RuntimeError: API key mancante: esporta GOOGLE_API_KEY nel tuo .env
```

**Soluzione**: Verifica che il file `.env` contenga la chiave API corretta.

#### **Memoria Insufficiente**

```
CUDA out of memory
```

**Soluzione**: Riduci la dimensione dei batch o usa CPU-only mode.

#### **PDF Non Processabili**

```
Failed to parse table HTML
```

**Soluzione**: Il sistema gestisce automaticamente i fallback per tabelle problematiche.

### **Log e Debug**

Il sistema fornisce output dettagliato durante l'elaborazione:

```
Processing PDF: ./pdfs/documento1.pdf
Captioning image: ./pdfs/documento1_p1.png
Text DB built with 150 documents.
Image DB built with 25 images.
```

## 🤝 Contributi

### **Struttura del Codice**

- `train_model.py`: Script principale con tutte le funzionalità
- `requirements.txt`: Dipendenze Python con versioni specifiche
- `README.md`: Documentazione completa

### **Estensioni Possibili**

- Supporto per altri formati di documento (DOCX, PPTX)
- Integrazione con altri modelli di embedding
- Interfaccia web con Streamlit/Gradio
- Supporto per database vettoriali alternativi
- Sistema di caching per ottimizzare le performance

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file LICENSE per i dettagli.

## 👨‍💻 Autore

**Stefano Leto** - Refactor luglio 2025

## 🙏 Ringraziamenti

- **LangChain**: Framework per RAG
- **Chroma**: Vector database
- **Google Gemini**: Modelli di linguaggio e visione
- **Hugging Face**: Modelli di embedding
- **OpenAI CLIP**: Modello di visione artificiale

---

**Versione**: v2025-07  
**Ultimo Aggiornamento**: Luglio 2025
