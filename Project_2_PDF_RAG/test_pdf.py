from pypdf import PdfReader

pdf_path = "data/Practical_Guide_to_Building_RAG.pdf"

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("\nFirst 2000 characters:\n")
print(text[:2000])