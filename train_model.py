#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multimodal RAG con comprensione immagini Gemini:
  • BGE-base-en-v1.5 (testo)
  • Gemini 2.5-flash (caption immagini + generazione)
  • Chroma vector store
Autore: Stefano Leto – 2025 (modificato)
"""
import os
import glob
import hashlib
import argparse
import shutil

# Disabilita telemetry
os.environ["LANGCHAIN_TELEMETRY_ENABLED"] = "false"
os.environ["CHROMA_DISABLE_TELEMETRY"]   = "1"

from dotenv import load_dotenv
load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

# ─── PDF ingest ──────────────────────────────────────────────────────────────
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── Embeddings testo ───────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings

# ─── Vector DB + LLM ─────────────────────────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

# ─── Gemini Image Understanding ──────────────────────────────────────────────
from google import genai
from google.genai import types

# Inizializza client Gemini
gi_client = genai.Client(api_key=GEMINI_KEY)

# ─── Helper md5 per mappa pagina ────────────────────────────────────────────
def md5h(txt: str) -> str:
    return hashlib.md5(txt.encode()[:120]).hexdigest()

# ─── Estrai testo + immagini dal PDF ────────────────────────────────────────
def load_pdf(pdf_path: str):
    el = partition_pdf(pdf_path, strategy="hi_res",
                       infer_table_structure=True,
                       skip_infer_table_types=False)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text("\n".join(e.text for e in el if e.text))

    pmap = {md5h(e.text): e.metadata.page_number for e in el if e.text}
    txt  = [
        {"text": c, "source": os.path.basename(pdf_path),
         "page": pmap.get(md5h(c), "-")}
        for c in chunks
    ]

    imgs = []
    import fitz
    from PIL import Image
    doc = fitz.open(pdf_path)
    for p in range(len(doc)):
        for idx, img in enumerate(doc.get_page_images(p), 1):
            xref = img[0]
            base = doc.extract_image(xref)
            ext  = base["ext"]
            path = f"{os.path.splitext(pdf_path)[0]}_p{p+1}_i{idx}.{ext}"
            with open(path, "wb") as f:
                f.write(base["image"])
            imgs.append({"path": path,
                         "source": os.path.basename(pdf_path),
                         "page": p+1})
    doc.close()
    return txt, imgs

# ─── Captioning immagini con Gemini 2.5 ────────────────────────────────────
def caption_image_with_gemini(path: str) -> str:
    with open(path, "rb") as f:
        img = f.read()
    mime = f"image/{path.split('.')[-1].lower().replace('jpg','jpeg')}"
    base_parts = [types.Part.from_bytes(data=img, mime_type=mime)]
    prompt = (
        "You are an engineering‐drawing assistant. "
        "1) List *all* dimension labels (e.g., l1…l5, r, D) you can read. "
        "2) Trascrivi eventuali relazioni (es. 'l5 = 0.5 × l4'). "
        "3) Fornisci una descrizione sintetica della forma.\n"
        "Return in Italian."
    )
    parts = base_parts + [types.Part.from_text(text=prompt)]
    resp = gi_client.models.generate_content(model="gemini-2.5-flash", contents=parts)
    return resp.text.strip()

# ─── Hybrid retriever ───────────────────────────────────────────────────────
class HybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    k: int = 10

    def _get_relevant_documents(self, query: str, **_) -> List[Document]:
        seen, docs = set(), []
        for r in self.retrievers:
            for d in r.get_relevant_documents(query):
                uid = (d.metadata.get("source"),
                       d.metadata.get("page"),
                       d.page_content[:60])
                if uid not in seen:
                    docs.append(d)
                    seen.add(uid)
        return docs[: self.k]

# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True,
                    help="Directory contenente i PDF da indicizzare")
    ap.add_argument("--persist_dir", default="./iso_index",
                    help="Directory di persistenza per Chroma")
    ap.add_argument("--k", type=int, default=10,
                    help="Numero di documenti da recuperare per query")
    args = ap.parse_args()

    # Pulisci indici esistenti
    for suf in ["_txt", "_img"]:
        shutil.rmtree(args.persist_dir + suf, ignore_errors=True)

    # Carica e processa PDF
    txt_docs, img_docs = [], []
    for pdf in glob.glob(os.path.join(args.pdf_dir, "*.pdf")):
        t, i = load_pdf(pdf)
        txt_docs.extend(t)
        img_docs.extend(i)

    # Crea caption per ogni immagine
    img_text_docs = []
    for img in img_docs:
        caption = caption_image_with_gemini(img["path"])
        img_text_docs.append({
            "text": caption,
            "source": img["source"],
            "page": img["page"]
        })

    # Unisci testo e caption immagini
    all_text_docs = txt_docs + img_text_docs

    # Embedding testo con BGE
    txt_embed = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vdb_txt = Chroma.from_texts(
        [d["text"] for d in all_text_docs],
        txt_embed,
        metadatas=[{"source": d["source"], "page": d["page"]}
                   for d in all_text_docs],
        persist_directory=args.persist_dir + "_txt"
    )

    # Imposta LLM Gemini per generazione risposte
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_KEY,
        temperature=0
    )

    # Prompt per QA
    qa_prompt = PromptTemplate(
        template=(
            "Use only the context below (text passages or image captions) "
            "to answer and cite file & page.\n\n{context}\n\nQ: {input}\nA:"
        ),
        input_variables=["context", "input"]
    )
    doc_prompt = PromptTemplate(
        template="{page_content}\n(Source: {source}, p.{page})",
        input_variables=["page_content", "source", "page"]
    )

    combine = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
        document_prompt=doc_prompt
    )
    hybrid = HybridRetriever(
        retrievers=[vdb_txt.as_retriever(search_kwargs={"k": args.k})],
        k=args.k
    )
    rag = create_retrieval_chain(
        retriever=hybrid,
        combine_docs_chain=combine
    )

    print("Multimodal RAG ready ➜ ask or blank to exit")
    while (q := input("\nDomanda › ").strip()):
        res = rag.invoke({"input": q})
        print("\n▸ Risposta:\n", res["answer"], "\n▸ Fonti:")
        for d in res["context"]:
            m = d.metadata
            print(f"  • {m.get('source','-')} (pag.{m.get('page','-')})")

if __name__ == "__main__":
    main()
