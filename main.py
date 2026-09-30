from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from explanation_module import explain_concept
from learning_path import get_learning_recommendations
from qna import answer_question
from quiz_module import generate_quiz
from summary_module import summarize_text


BASE_DIR = Path(__file__).resolve().parent


app = FastAPI(
    title="EduGenie",
    description="Google Gemini Powered Learning Assistant",
    version="1.0.0",
)


app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=15000,
        description="User input text",
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Display the EduGenie web interface.
    """

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "EduGenie - Learning Assistant"
        },
    )


@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """

    return {
        "status": "ok",
        "application": "EduGenie",
    }


@app.post("/qa")
def qa_endpoint(payload: TextRequest):
    """
    Question answering endpoint.
    """

    try:
        result = answer_question(payload.text)

        return {
            "success": True,
            "type": "qa",
            "result": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/explain")
def explain_endpoint(payload: TextRequest):
    """
    Concept explanation endpoint.
    """

    try:
        result = explain_concept(payload.text)

        return {
            "success": True,
            "type": "explain",
            "result": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/quiz")
def quiz_endpoint(payload: TextRequest):
    """
    Quiz generation endpoint.
    """

    try:
        result = generate_quiz(payload.text)

        return {
            "success": True,
            "type": "quiz",
            "result": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/summarize")
def summarize_endpoint(payload: TextRequest):
    """
    Summarization endpoint.
    """

    try:
        result = summarize_text(payload.text)

        return {
            "success": True,
            "type": "summary",
            "result": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/learn/recommendations")
def learning_recommendations_endpoint(payload: TextRequest):
    """
    Personalized learning path endpoint.
    """

    try:
        result = get_learning_recommendations(payload.text)

        return {
            "success": True,
            "type": "learning_path",
            "result": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc