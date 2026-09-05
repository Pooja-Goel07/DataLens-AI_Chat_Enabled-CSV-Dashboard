# documents.py - Upload / list / delete documents for the RAG knowledge base.
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pypdf import PdfReader

from auth import get_current_user
import db
import rag

documents_router = APIRouter()


def _extract_text(filename: str, contents: bytes) -> str:
    """Extract plain text from a PDF or TXT upload."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(contents))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".txt"):
        return contents.decode("utf-8", errors="ignore")
    raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")


@documents_router.post("/upload_document/")
async def upload_document(
    file: UploadFile = File(...),
    doc_name: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    """Ingest a document into the user's knowledge base."""
    contents = await file.read()
    text = _extract_text(file.filename, contents)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in the document.")

    try:
        # Replace any previous version of this document (vectors + metadata) before re-ingesting.
        rag.delete_document(current_user, doc_name)
        chunks = rag.ingest_document(current_user, doc_name, text)
        if chunks == 0:
            raise HTTPException(status_code=400, detail="Document produced no usable text chunks.")
        db.record_document(current_user, doc_name, file.filename, chunks)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    return {
        "message": f"Document '{doc_name}' ingested successfully.",
        "doc_name": doc_name,
        "file_name": file.filename,
        "chunks": chunks,
        "success": True,
    }


@documents_router.get("/list_documents/")
async def list_documents(current_user: str = Depends(get_current_user)):
    """List the current user's documents."""
    return {"documents": db.list_user_documents(current_user), "user": current_user}


@documents_router.delete("/document/{doc_name}")
async def delete_document(doc_name: str, current_user: str = Depends(get_current_user)):
    """Delete a document (vectors + metadata) owned by the user."""
    existed = db.delete_document_record(current_user, doc_name)
    if not existed:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
    rag.delete_document(current_user, doc_name)
    return {"message": f"Document '{doc_name}' deleted.", "deleted_document": doc_name}
