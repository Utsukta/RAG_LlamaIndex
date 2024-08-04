# convert the document to markdown from pymupdf4llm
import pymupdf4llm
# md_text = pymupdf4llm.to_markdown("Proposal-ict.pdf")
# print(md_text)
file_path="/Users/utsuktakhatri/RAG_LlamaIndex/data"
a=pymupdf4llm.LlamaMarkdownReader().load_data("Proposal-ict.pdf")
print(a)

# from unstructured.partition.pdf import partition_pdf

# # elements = partition_pdf(filename="proposal-ict.pdf")

# from unstructured.partition.auto import partition
# elements = partition(filename="Proposal-ict.pdf")
# print("\n\n".join([str(el) for el in elements]))

# from unstructured.partition.auto import partition

# filename = "sales.pdf"

# elements = partition(filename=filename,
#                      strategy='hi_res',
#            )

# tables = [el for el in elements if el.category == "Table"]
# print(tables)

# print(tables[0].text)
# print(tables[0].metadata.text_as_html)


###Unstrcutured
# from unstructured.partition.pdf import partition_pdf

# fname = "Proposal-ict.pdf"

# elements = partition_pdf(filename=fname,
#                          infer_table_structure=True,
#                          strategy='hi_res',
#            )

# tables = [el for el in elements if el.category == "Table"]
# for row in range(len(tables)):
#     print(tables[row].text)

# print("heloooooooooooooooo")

# import pdfplumber

# with pdfplumber.open("Proposal-ict.pdf") as pdf:
#     first_page = pdf.pages[7]
#     # print(first_page.chars[0])
#     print(first_page.extract_tables())

