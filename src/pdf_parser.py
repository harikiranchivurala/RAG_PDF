from pdfminer.high_level import extract_text

class PDFParser:
    """Extracts text from a PDF document."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extracts raw text from the given PDF file.
        
        Args:
            pdf_path (str): The path to the PDF file.
        
        Returns:
            str: The extracted text from the PDF.
        """
        try:
            text = extract_text(pdf_path)
            # print(text)
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")


# PDFParser.extract_text_from_pdf("data/zania_handbook.pdf")