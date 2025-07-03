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
    """Restituisce (chunks_testo, infos_immagini)"""
    elements = partition_pdf(pdf_path, strategy="hi_res", infer_table_structure=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text("\n".join(e.text for e in elements if e.text))

    pmap = {md5h(e.text): e.metadata.page_number for e in elements if e.text}
    txt_docs = [{
        "text": c,
        "source": os.path.basename(pdf_path),
        "page": pmap.get(md5h(c), "-")
    } for c in chunks]

    import fitz  # PyMuPDF
    img_infos = []
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        # rasterizza l'intera pagina a 300 dpi (leggibilità quote)
        pix = page.get_pixmap(dpi=300)  # chiave: alta risoluzione
        img_path = f"{os.path.splitext(pdf_path)[0]}_p{page_index+1}.png"
        pix.save(img_path)
        img_infos.append({
            "path": img_path,
            "source": os.path.basename(pdf_path),
            "page": page_index + 1
        })
    doc.close()
    return txt_docs, img_infos

# ── Caption immagini con Gemini (JSON) ─────────────────────────────────────

def caption_image_with_gemini(path: str) -> str:
    with open(path, "rb") as f:
        img_bytes = f.read()
    mime = f"image/{path.split('.')[-1].lower().replace('jpg','jpeg')}"

    prompt = (
        "Leggi il disegno tecnico e restituisci un oggetto JSON con 3 campi:\n"
        "labels: elenco di tutte le etichette di quota (es. ['l1', 'l2', 'r']).\n"
        "relations: elenco di relazioni testuali trovate (es. 'l5 = 0,5 × l4').\n"
        "shape: breve descrizione in italiano (<30 parole)."
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
        return "{}"  # fallback JSON vuoto

# ── Visual Question Answering (fallback) ───────────────────────────────────

def ask_gemini_vqa(img_path: str, question: str) -> str:
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    mime = f"image/{img_path.split('.')[-1].lower().replace('jpg','jpeg')}"
    parts = [
        types.Part.from_bytes(data=img_bytes, mime_type=mime),
        types.Part.from_text(text=question)
    ]
    resp = gi_client.models.generate_content(model="gemini-2.5-flash", contents=parts)
    return resp.text.strip()

# ── Hybrid retriever ───────────────────────────────────────────────────────
class HybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    k: int = 10

    def _get_relevant_documents(self, query: str, **_) -> List[Document]:
        seen, docs = set(), []
        for r in self.retrievers:
            # .invoke sostituisce get_relevant_documents da LC 0.1.46
            hits = r.invoke(query) if hasattr(r, "invoke") else r.get_relevant_documents(query)
            for d in hits:
                uid = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:60])
                if uid not in seen:
                    docs.append(d)
                    seen.add(uid)
        return docs[: self.k]

# ── CLI ────────────────────────────────────────────────────────────────────

def build_index(pdf_dir: str, persist_dir: str, k: int):
    # Pulizia
    for suf in ["_txt", "_img"]:
        shutil.rmtree(persist_dir + suf, ignore_errors=True)

    txt_docs, img_infos = [], []
    for pdf in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        t, i = load_pdf(pdf)
        txt_docs.extend(t)
        img_infos.extend(i)

    # Caption + JSON
    img_text_docs = []
    for info in img_infos:
        caption = caption_image_with_gemini(info["path"])
        img_text_docs.append({
            "text": caption,
            "source": info["source"],
            "page": info["page"],
            "path": info["path"]  # per VQA fallback
        })

    # Vector store testo (PDF + caption)
    all_text_docs = txt_docs + img_text_docs
    txt_embed = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vdb_txt = Chroma.from_texts(
        [d["text"] for d in all_text_docs],
        txt_embed,
        metadatas=[{"source": d["source"], "page": d["page"], "path": d.get("path")}
                   for d in all_text_docs],
        persist_directory=persist_dir + "_txt",
    )

    # Vector store immagini (CLIP)
    clip_embed = CLIPEmbeddings()
    vdb_img = Chroma.from_texts(
        [i["path"] for i in img_infos],
        embedding=clip_embed,
        metadatas=img_infos,
        persist_directory=persist_dir + "_img",
    )

    return vdb_txt, vdb_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Cartella PDF")
    ap.add_argument("--persist_dir", default="./iso_index", help="Path indice Chroma")
    ap.add_argument("--k", type=int, default=10, help="Top‑k retrieval")
    args = ap.parse_args()

    vdb_txt, vdb_img = build_index(pdf_dir=args.pdf_dir, persist_dir=args.persist_dir, k=args.k)

    # LLM Gemini per le risposte
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_KEY, temperature=0)

    # Prompting chain
    qa_prompt = PromptTemplate(
        template=(
            "Usa solo il contesto (passaggi testuali o caption) per rispondere. "
            "Cita file & pagina.\n\n{context}\n\nQ: {input}\nA:"),
        input_variables=["context", "input"])
    doc_prompt = PromptTemplate(
        template="{page_content}\n(Source: {source}, p.{page})",
        input_variables=["page_content", "source", "page"])

    combine_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt, document_prompt=doc_prompt)

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

        # 1. standard RAG
        res = rag.invoke({"input": q})
        answer = res["answer"].strip()

        # 2. fallback VQA se la risposta menziona "non" + immagini CLIP correlate
        if "non" in answer.lower():
            clip_hits: List[Document] = vdb_img.similarity_search(query=q, k=2)
            for doc in clip_hits:
                vqa_ans = ask_gemini_vqa(doc.metadata["path"], q)
                if vqa_ans and "non posso" not in vqa_ans.lower():
                    answer = vqa_ans + f"\n\n(Source VQA: {doc.metadata['source']} p.{doc.metadata['page']})"
                    res["context"].append(doc)
                    break

        # Output
        print("\n▸ Risposta:\n", answer, "\n▸ Fonti:")
        for d in res["context"]:
            m = d.metadata
            print(f"  • {m.get('source','-')} (pag.{m.get('page','-')})")


if __name__ == "__main__":
    main()
