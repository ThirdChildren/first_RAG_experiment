#!/usr/bin/env python
# RAG Gemini 2.5 Flash + Chroma – LCEL pipeline
import os, glob, argparse
os.environ["LANGCHAIN_TELEMETRY_ENABLED"] = "false"

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from unstructured.partition.pdf import partition_pdf

# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_dir",    required=True, help="Folder with PDFs")
    p.add_argument("--persist_dir",default="./iso_index", help="Chroma persistence")
    p.add_argument("--k",          type=int, default=10, help="Top-k docs")
    return p.parse_args()

# ---------- PDF ➜ chunks (+OCR) ----------
def load_and_chunk(pdf_path):
    elements = partition_pdf(
        pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        skip_infer_table_types=False,
    )
    text_all = "\n".join(e.text for e in elements if e.text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text_all)
    # mappa hash→pagina
    page_map = {hash(el.text[:120]): el.metadata.page_number
                for el in elements if el.text}
    return [
        {
            "text": ch,
            "source": os.path.basename(pdf_path),
            "page": page_map.get(hash(ch[:120]), "-"),
        }
        for ch in chunks
    ]

# ---------- MAIN ----------
def main():
    args = get_args()

    # 1) ingest
    docs = []
    for pdf in glob.glob(os.path.join(args.pdf_dir, "*.pdf")):
        docs.extend(load_and_chunk(pdf))

    # 2) embeddings + Chroma
    embed = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    vectordb = Chroma.from_texts(
        texts=[d["text"] for d in docs],
        embedding=embed,
        metadatas=[{"source":d["source"],"page":d["page"]} for d in docs],
        persist_directory=args.persist_dir
    )

    # 3) LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    # 4) Retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": args.k})

    # 5) Prompt templates (EN)
    qa_prompt = PromptTemplate(
        template=(
            "Answer the user question based on the following context:\n\n"
            "{context}\n\n"
            "Question: {input}\n"
            "Answer:"
        ),
        input_variables=["context", "input"],
    )
    doc_prompt = PromptTemplate(
        template="Passage:\n{page_content}\n(Source: {source}, Page: {page})",
        input_variables=["page_content", "source", "page"],
    )

    # 6) Combine documents chain
    combine_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,               # must accept 'context'
        document_prompt=doc_prompt,     # formats each doc
    )

    # 7) Retrieval chain
    qa_chain = create_retrieval_chain(retriever, combine_chain)

    # 8) Loop interattivo
    print("RAG ready! Enter a question or blank to exit.")
    while (q := input("\nDomanda › ").strip()):
        res = qa_chain.invoke({"input": q})
        answer = res.get("answer") or res.get("result","")
        docs_out = res.get("context") or res.get("documents",[])
        print("\n▸ Risposta:\n", answer, "\n")
        print("▸ Fonti:")
        for d in docs_out:
            meta = d.metadata
            print(f"  • {meta.get('source','-')} (pag. {meta.get('page','-')})")

if __name__ == "__main__":
    main()
