
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter



def chunk(file:str):
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0,
    )
    md_header_splits = []
    with open(file, 'r') as file:
        content = file.read()
        md_header_splits = markdown_splitter.split_text(content)
    res = []
    for doc in md_header_splits:
        chunks = splitter.split_text(doc.page_content)
        heading = ""
        if "h1" in doc.metadata:
            heading = doc.metadata["h1"]
        if "h2" in doc.metadata:
            heading = doc.metadata["h2"]
        if "h3" in doc.metadata:
            heading = doc.metadata["h3"]
        if "h4" in doc.metadata:
            heading = doc.metadata["h4"]
        res.append({
            "heading":heading,
            "chunks":chunks,
        })
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    

chunk("MinerU_2307.09288v2__20251127030211.md")