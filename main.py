from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from profile_processing_script import  data_processing
from fastapi.responses import JSONResponse

app = FastAPI()

app.mount("/resumes_corpus", 
        StaticFiles(directory="/Users/uday.shaw/Documents/private/ISB/term2/FP1/project_impl/dataset/resumes_corpus"), 
        name="resumes_corpus")

templates = Jinja2Templates(directory="templates")

class File(BaseModel):
    name: str
    description: str | None = None
    

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_text(request: Request):
    data = await request.json()
    #text = data.get("text")
    # Perform text analysis or any other processing here
    # For demonstration purposes, let's simply return the uppercase version of the text
    #return {"result": text.upper()}
    decoded_contents = data.get("text")
    dp=data_processing()
    text=dp.get_profiles(decoded_contents)
    return JSONResponse(content={"message": text})


@app.get("/add_profile", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("add_profile.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile):
    contents = await file.read()
    decoded_contents = contents.decode("utf-8", errors="ignore")
    dp=data_processing()
    text=dp.process_content(file.filename, decoded_contents)
    return JSONResponse(content={"message": text})

