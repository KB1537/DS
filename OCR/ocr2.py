from PyMuPDF import fitz # PyMuPDF
 

def convert_pdf_to_text(pdf_file):
    text_content = ""
    with fitz.open(pdf_file) as pdf_document:
        num_pages = pdf_document.page_count
 
        for page_number in range(num_pages):
            page = pdf_document[page_number]
            text_content += page.get_text()
 
    return text_content
 

if __name__ == "__main__":
    pdf_file = "Data Analyst 1.1 Apprentice Toolkit v2.1-22.pdf"
    output_text = convert_pdf_to_text(pdf_file)
 
    # Print or save the extracted text
    print(output_text.encode("utf-8"))
 
    # If you want to save the text to a file
    with open("output.txt", "w", encoding="utf-8") as output_file:
        output_file.write(output_text)
 