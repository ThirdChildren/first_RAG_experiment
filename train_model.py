#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multimodal RAG:
  • BGE-base-en-v1.5 (testo, locale)
  • CLIP ViT-B-32 (immagini)
  • Chroma vector store
  • Gemini Flash 2.5 per generazione risposta
Autore: Stefano Leto – 2025
"""
import os, glob, hashlib, argparse, shutil
os.environ["LANGCHAIN_TELEMETRY_ENABLED"] = "false"
os.environ["CHROMA_DISABLE_TELEMETRY"]   = "1"

from dotenv import load_dotenv
load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

# ─── PDF ingest ──────────────────────────────────────────────────────────────
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── Embeddings locali ───────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
import torch, open_clip, fitz
from PIL import Image
from langchain_core.embeddings import Embeddings

# ─── Vector DB + LLM ─────────────────────────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

""" from transformers import AutoProcessor, Blip2ForConditionalGeneration
caption_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
caption_proc  = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b") """

# ─── Helper md5 per mappa pagina ────────────────────────────────────────────
def md5h(txt: str) -> str: return hashlib.md5(txt.encode()[:120]).hexdigest()

# ─── Estrai testo + immagini dal PDF ────────────────────────────────────────
def load_pdf(pdf_path: str):
    el = partition_pdf(pdf_path, strategy="hi_res",
                       infer_table_structure=True, skip_infer_table_types=False)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks   = splitter.split_text("\n".join(e.text for e in el if e.text))

    pmap = {md5h(e.text): e.metadata.page_number for e in el if e.text}
    txt  = [ {"text":c, "source":os.path.basename(pdf_path),
              "page":pmap.get(md5h(c),"-")} for c in chunks ]

    imgs = []
    doc = fitz.open(pdf_path)
    for p in range(len(doc)):
        for idx, img in enumerate(doc.get_page_images(p), 1):
            xref = img[0]; base = doc.extract_image(xref)
            path = f"{os.path.splitext(pdf_path)[0]}_p{p+1}_i{idx}.{base['ext']}"
            with open(path,"wb") as f: f.write(base["image"])
            imgs.append({"path":path,"source":os.path.basename(pdf_path),"page":p+1})
    doc.close()
    return txt, imgs

# ─── CLIP embedding wrapper ──────────────────────────────────────────────────
# ← sostituisci la vecchia classe
class CLIPEmbeddings(Embeddings):
    def __init__(self):
        # model, tokenizer, preprocess
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"   # ← device
        self.model = self.model.to(self.device).eval()

    # immagini → embeddings
    def embed_documents(self, paths):
        out = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            with torch.no_grad():
                vec = self.model.encode_image(
                    self.pre(img).unsqueeze(0).to(self.device)
                ).squeeze().tolist()
            out.append(vec)
        return out

    # testo → embeddings
    def embed_query(self, query: str):
        with torch.no_grad():
            tok = self.tokenizer([query]).to(self.device)   # ← lista + device
            vec = self.model.encode_text(tok).squeeze().tolist()
        return vec


# ─── Hybrid retriever (parallelo) ────────────────────────────────────────────
class HybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]        
    k: int = 10

    def _get_relevant_documents(self, query: str, **_) -> List[Document]:
        seen, docs = set(), []
        for r in self.retrievers:
            for d in r.get_relevant_documents(query):
                uid = (d.metadata.get("source"), d.metadata.get("page"),
                       d.page_content[:60])
                if uid not in seen:
                    docs.append(d); seen.add(uid)
        return docs[: self.k]

# ─── CLI ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--pdf_dir", required=True)
ap.add_argument("--persist_dir", default="./iso_index")
ap.add_argument("--k", type=int, default=10)
args = ap.parse_args()

# ─── Rebuild index ───────────────────────────────────────────────────────────
for suf in ["_txt","_img"]: shutil.rmtree(args.persist_dir+suf, ignore_errors=True)

txt_docs, img_docs = [], []
for pdf in glob.glob(os.path.join(args.pdf_dir,"*.pdf")):
    t,i = load_pdf(pdf); txt_docs.extend(t); img_docs.extend(i)

txt_embed = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5",
                                  encode_kwargs={"normalize_embeddings":True})
vdb_txt = Chroma.from_texts([d["text"] for d in txt_docs], txt_embed,
         metadatas=[{"source":d["source"],"page":d["page"]} for d in txt_docs],
         persist_directory=args.persist_dir+"_txt")

img_embed = CLIPEmbeddings()
vdb_img  = Chroma.from_texts([d["path"] for d in img_docs], img_embed,
         metadatas=[{"source":d["source"],"page":d["page"],"path":d["path"]}
                    for d in img_docs],
         persist_directory=args.persist_dir+"_img")

hybrid = HybridRetriever(
    retrievers=[
        vdb_txt.as_retriever(search_kwargs={"k": args.k}),
        vdb_img.as_retriever(search_kwargs={"k": args.k}),
    ],
    k=args.k
)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                             google_api_key=GEMINI_KEY, temperature=0)

qa_prompt = PromptTemplate(
    template=("Use only the context below (text passages or image captions) "
              "to answer and cite file & page.\n\n{context}\n\nQ: {input}\nA:"),
    input_variables=["context","input"])
doc_prompt = PromptTemplate(
    template="{page_content}\n(Source: {source}, p.{page})",
    input_variables=["page_content","source","page"])

combine = create_stuff_documents_chain(llm=llm,
                                      prompt=qa_prompt,
                                      document_prompt=doc_prompt)
rag     = create_retrieval_chain(retriever=hybrid,
                                 combine_docs_chain=combine)

print("Multimodal RAG ready ➜ ask or blank to exit")
while (q := input("\nDomanda › ").strip()):
    res  = rag.invoke({"input": q})
    print("\n▸ Risposta:\n", res["answer"], "\n▸ Fonti:")
    for d in res["context"]:
        m=d.metadata; print(f"  • {m.get('source','-')} (pag.{m.get('page','-')})")
