#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multimodal RAG – v2025‑07
  • Testo  : BGE‑base‑en‑v1.5
  • Immagini: Gemini 2.5‑flash (caption + VQA) ✚ CLIP ViT‑B‑32 (per recall)
  • Vector  : Chroma
Autore: Stefano Leto – refactor luglio 2025
"""
import os, glob, hashlib, argparse, shutil, json, warnings
import pandas as pd 
import io # Importa io per StringIO, utile per pandas.read_html
from typing import List, Optional

# ── Disabilita telemetria LangChain ────────────────────────────────────────
os.environ.setdefault("LANGCHAIN_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CHROMA_DISABLE_TELEMETRY", "1")

from dotenv import load_dotenv
load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("API key mancante: esporta GOOGLE_API_KEY nel tuo .env")

# ── PDF ingest ─────────────────────────────────────────────────────────────
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Embeddings testo + immagini ────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings

# CLIP per recall visivo
import torch, open_clip
from PIL import Image
from langchain_core.embeddings import Embeddings


class CLIPEmbeddings(Embeddings):
    """Wrapper minimale su CLIP ViT‑B‑32 (OpenAI)"""

    def __init__(self):
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    # Un testo → embedding (per query)
    def embed_query(self, text: str):
        with torch.no_grad():
            tok = self.tokenizer([text]).to(self.device)
            return self.model.encode_text(tok).squeeze().tolist()

    # Una lista di path image → embeddings
    def embed_documents(self, paths: List[str]):
        out = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            with torch.no_grad():
                vec = self.model.encode_image(
                    self.pre(img).unsqueeze(0).to(self.device)
                ).squeeze().tolist()
            out.append(vec)
        return out

# ── Vector DB & LLM ────────────────────────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# ── Gemini Image Understanding ─────────────────────────────────────────────
from google import genai
from google.genai import types

gi_client = genai.Client(api_key=GEMINI_KEY)

# ── Helper ────────────────────────────────────────────────────────────────
def md5h(txt: str) -> str:
    return hashlib.md5(txt.encode()[:120]).hexdigest()

# ── Estrattore PDF (testo + immagini in alta risoluzione) ─────────────────

def load_pdf(pdf_path: str):
    """Restituisce (chunks_testo, infos_immagini) con metadati più ricchi."""
    
    elements = partition_pdf(
        pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
    )

    docs_from_elements = []
    for e in elements:
        if e.text:
            metadata = {
                "source": os.path.basename(pdf_path),
                "page": e.metadata.page_number,
                "type": e.category if hasattr(e, 'category') else str(type(e)).split('.')[-1].replace("'>", ""), 
            }
            
            # Gestione robusta dell'header
            header_str = ""
            if hasattr(e.metadata, 'ancestor_paths') and e.metadata.ancestor_paths:
                header_str = " > ".join(e.metadata.ancestor_paths)
            elif metadata["type"] == "Title" and e.text: # Se è un titolo e ha testo, usiamo il suo testo
                header_str = e.text
            metadata["header"] = header_str if header_str else None # Assicurati che 'header' sia sempre presente, anche se None
            
            # Formatta la stringa completa delle informazioni sulla fonte
            source_info = metadata.get('source', '-')
            page_info = metadata.get('page', '-')
            type_info = f", Type: {metadata['type']}" if metadata.get('type') else ""
            header_for_source_info = f", Section: {metadata['header']}" if metadata.get('header') else ""
            
            # Aggiungi 'source_info_full' direttamente ai metadati
            metadata["source_info_full"] = f"{source_info}, p.{page_info}{header_for_source_info}{type_info}"
            
            if e.metadata.text_as_html:
                try:
                    # Usa StringIO per passare l'HTML come una stringa-file a read_html
                    df = pd.read_html(io.StringIO(e.metadata.text_as_html))[0]
                    table_text = "Tabella:\n" + df.to_markdown(index=False)
                    docs_from_elements.append(Document(page_content=table_text, metadata=metadata))
                except Exception as ex:
                    warnings.warn(f"Failed to parse table HTML from {pdf_path} (page {e.metadata.page_number}): {ex}")
                    docs_from_elements.append(Document(page_content=e.text, metadata=metadata)) # Fallback
            else:
                docs_from_elements.append(Document(page_content=e.text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    final_txt_chunks = splitter.split_documents(docs_from_elements)

    txt_docs = []
    for doc in final_txt_chunks:
        new_metadata = doc.metadata.copy()
        new_metadata["text_chunk_hash"] = md5h(doc.page_content) 
        txt_docs.append({
            "text": doc.page_content,
            **new_metadata
        })

    import fitz  # PyMuPDF
    img_infos = []
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img_path = f"{os.path.splitext(pdf_path)[0]}_p{page_index+1}.png"
        pix.save(img_path)
        img_infos.append({
            "path": img_path,
            "source": os.path.basename(pdf_path),
            "page": page_index + 1
        })
    doc.close()
    return txt_docs, img_infos

# ── Caption immagini con Gemini (Descrizione generale migliorata) ─────────────────────────────────────

def caption_image_with_gemini(path: str) -> str:
    with open(path, "rb") as f:
        img_bytes = f.read()
    mime = f"image/{path.split('.')[-1].lower().replace('jpg','jpeg')}"

    # PROMPT MIGLIORATO PER PIÙ DETTAGLI E STRUTTURA
    prompt = (
        "Sei un assistente specializzato nella descrizione dettagliata di immagini tecniche e scientifiche. "
        "Fornisci una descrizione *esaustiva* e *strutturata* del contenuto di questa immagine. "
        "Includi la didascalia originale dell'immagine se presente e distinguila dalla descrizione generata. "
        "Identifica e descrivi tutti gli elementi principali e secondari visibili, i loro attributi "
        "(es. forma, dimensioni, colore, materiale se implicito) e le loro relazioni spaziali. "
        "Se l'immagine è un disegno tecnico, un diagramma, una tabella o un grafico, estrai i seguenti dettagli:\n"
        "- **Titolo/Didascalia Implicita:** Se non c'è una didascalia esplicita, inferiscila dal contenuto visivo.\n"
        "- **Elementi Chiave:** Elenca gli oggetti principali con i loro nomi (se visibili) e il loro ruolo.\n"
        "- **Dettagli Numerici:** Riporta tutti i valori, quote, percentuali, intervalli di dati e unità di misura visibili.\n"
        "- **Relazioni/Struttura:** Descrivi le connessioni, le gerarchie, le tendenze, i processi o il layout illustrati tra gli elementi.\n"
        "- **Orientamento/Prospettiva:** Indica l'angolo di visione o la vista rappresentata (es. vista frontale, vista dall'alto, assonometrica).\n"
        "Sii preciso, completo e organizzato usando elenchi puntati o numerati dove appropriato per facilitare la lettura. "
        "Rispondi *solo* in italiano e non fare riferimento al fatto che stai descrivendo un'immagine, bensì il suo contenuto diretto."
    )

    parts = [
        types.Part.from_bytes(data=img_bytes, mime_type=mime),
        types.Part.from_text(text=prompt)
    ]
    try:
        resp = gi_client.models.generate_content(model="gemini-2.5-flash", contents=parts)
        return resp.text.strip()
    except Exception as e:
        warnings.warn(f"Gemini caption failed on {path}: {e}")
        return f"Descrizione non disponibile per l'immagine {os.path.basename(path)}. Errore: {e}"

# ── Visual Question Answering (Funzione mantenuta ma fallback automatico rimosso) ───────────────────────────────────

# ── Hybrid retriever (Aumento k e metadati più ricchi) ───────────────────────────────────────────────────────
class HybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    k: int = 15 

    def _get_relevant_documents(self, query: str, **_) -> List[Document]:
        seen, docs = set(), []
        for r in self.retrievers:
            hits = r.invoke(query) if hasattr(r, "invoke") else r.get_relevant_documents(query)
            for d in hits:
                uid = (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("text_chunk_hash", md5h(d.page_content[:60])))
                if uid not in seen:
                    docs.append(d)
                    seen.add(uid)
        return docs[: self.k]

# ── CLI ────────────────────────────────────────────────────────────────────

def build_index(pdf_dir: str, persist_dir: str, k: int):
    for suf in ["_txt", "_img"]:
        shutil.rmtree(persist_dir + suf, ignore_errors=True)

    txt_docs, img_infos = [], []
    for pdf in glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True): 
        print(f"Processing PDF: {pdf}")
        t, i = load_pdf(pdf)
        txt_docs.extend(t)
        img_infos.extend(i)

    img_text_docs = []
    for info in img_infos:
        print(f"Captioning image: {info['path']}")
        caption = caption_image_with_gemini(info["path"])
        
        # Aggiungi source_info_full anche per le caption delle immagini
        source_info = info.get('source', '-')
        page_info = info.get('page', '-')
        # Le caption non hanno header o type specifici come testo del PDF, quindi li omettiamo qui
        info["source_info_full"] = f"{source_info}, p.{page_info}, Type: Image Caption"
        
        info_with_caption_hash = info.copy()
        info_with_caption_hash["text_chunk_hash"] = md5h(caption)
        img_text_docs.append({
            "text": caption,
            **info_with_caption_hash
        })

    all_text_docs = txt_docs + img_text_docs
    txt_embed = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vdb_txt = Chroma.from_texts(
        [d["text"] for d in all_text_docs],
        txt_embed,
        metadatas=[{k: v for k, v in d.items() if k != "text"}
                   for d in all_text_docs],
        persist_directory=persist_dir + "_txt",
    )
    print(f"Text DB built with {len(all_text_docs)} documents.")

    clip_embed = CLIPEmbeddings()
    vdb_img = Chroma.from_texts(
        [i["path"] for i in img_infos],
        embedding=clip_embed,
        metadatas=img_infos,
        persist_directory=persist_dir + "_img",
    )
    print(f"Image DB built with {len(img_infos)} images.") 

    return vdb_txt, vdb_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Cartella PDF")
    ap.add_argument("--persist_dir", default="./iso_index", help="Path indice Chroma")
    ap.add_argument("--k", type=int, default=15, help="Top‑k retrieval (documenti totali)") 
    args = ap.parse_args()

    vdb_txt, vdb_img = build_index(pdf_dir=args.pdf_dir, persist_dir=args.persist_dir, k=args.k)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_KEY, temperature=0.1) 

    qa_prompt = PromptTemplate(
        template=(
            "Sei un assistente esperto in standard tecnici e studi scientifici sui parastinchi. "
            "Rispondi alla domanda usando *esclusivamente* il contesto fornito. "
            "Sii il più esaustivo possibile nell'estrarre i dettagli pertinenti. "
            "Se la risposta non è presente nel contesto, o se il contesto è insufficiente per rispondere in modo completo, "
            "dichiara chiaramente 'Le informazioni richieste non sono disponibili nel contesto fornito in modo completo.' "
            "Cita sempre il nome del file e il numero di pagina per ogni informazione estratta dal contesto. "
            "Per le tabelle o elenchi puntati, estrai tutti i dettagli rilevanti e presentali in un formato chiaro (es. elenco puntato o testo continuo). \n\n"
            "Contesto:\n{context}\n\n"
            "Domanda: {input}\n"
            "Risposta esaustiva e citata:"),
        input_variables=["context", "input"])
    
    document_prompt = PromptTemplate(
        template="{page_content}\n(Source: {source_info_full})",
        input_variables=["page_content", "source_info_full"]
    )

    combine_chain = create_stuff_documents_chain(
        llm=llm, 
        prompt=qa_prompt, 
        document_prompt=document_prompt 
    )
    
    hybrid = HybridRetriever(retrievers=[
        vdb_txt.as_retriever(search_kwargs={"k": args.k}),
        vdb_img.as_retriever(search_kwargs={"k": 3})
    ], k=args.k)

    rag = create_retrieval_chain(retriever=hybrid, combine_docs_chain=combine_chain)

    print("Multimodal RAG pronto → premi Invio vuoto per uscire")
    while True:
        q = input("\nDomanda › ").strip()
        if not q:
            break

        res = rag.invoke({"input": q})
        answer = res["answer"].strip()

        # Output delle fonti: estrai 'source_info_full' direttamente dai metadati
        print("\n▸ Risposta:\n", answer, "\n▸ Fonti:")
        for d in res["context"]:
            source_info_full = d.metadata.get('source_info_full', f"{d.metadata.get('source', '-')}, p.{d.metadata.get('page', '-')}")
            print(f"  • {d.page_content[:100]}... (Source: {source_info_full})")


if __name__ == "__main__":
    main()