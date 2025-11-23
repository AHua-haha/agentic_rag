


import pathlib
import pymupdf4llm
import pprint

# md_text = pymupdf4llm.to_markdown(
#     doc="2307.09288v2.pdf",
#     write_images=True,
#     ignore_graphics=True,
#     image_path="./img/",
#     table_strategy="lines"
# )

# pathlib.Path("output.md").write_bytes(md_text.encode())

from langchain_text_splitters import MarkdownHeaderTextSplitter




headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
with open('output.md', 'r') as file:
    content = file.read()
    md_header_splits = markdown_splitter.split_text(content)
    print(md_header_splits)