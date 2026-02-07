from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import json
import tempfile
import os
import uvicorn
from agent import analyze_person
from dnd_pdf_filler_simple.generate_character import generate_character_sheet

app = FastAPI()

class Request(BaseModel):
    description: str

@app.get("/", response_class=HTMLResponse)
async def home():
    return open("index.html").read()

@app.post("/analyze")
async def analyze(req: Request):
    result = analyze_person(req.description)
    #here is json

    try:
        character = json.loads(result)
    except json.JSONDecodeError:
        return {"error": "Failed to parse character sheet", "raw": result}
    try:
        character = json.loads(result)
    except json.JSONDecodeError:
        return {"error": "Failed to parse character sheet", "raw": result}

    # 2. Save JSON to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(character, f)
        temp_json_path = f.name

    # 3. Generate PDF
    try:
        pdf_path = generate_character_sheet(temp_json_path, output_folder="/tmp/sheets")
        return FileResponse(pdf_path, filename=os.path.basename(pdf_path), media_type="application/pdf")
    finally:
        os.unlink(temp_json_path)
    #return {"character_sheet": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)