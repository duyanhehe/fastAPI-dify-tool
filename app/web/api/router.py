import os

from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from fastapi import File, UploadFile, HTTPException

from app.core.settings import settings
from app.repositories.generate_text import (
    extract_text_from_file,
    generate_questions,
    save_output,
)
from app.repositories.generate_letter_of_recommendation import letter_of_recommendation
from app.repositories.questions_from_YT import *
from app.schemas.request_schema import GenerateTextRequest
from app.utils.api_utils import make_response
from app.web.api import echo, monitoring

from typing import List
from pathlib import Path
from io import BytesIO
import os

api_router = APIRouter()
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(echo.router, prefix="/echo", tags=["echo"])


ALLOWED_EXTENSIONS = {"pdf", "docx", "pptx"}


def allowed_files(filename: str) -> bool:
    """Check if the file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@api_router.post("/generate_text")
async def generate_text(
    file: UploadFile = File(...),
    num_questions: int = 5,
    question_type: str = "multiple-choice",
    output_format: str = "markdown",
) -> StreamingResponse:
    try:
        # Check if the uploaded file is allowed
        if not allowed_files(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only PDF, DOCX, and PPTX are allowed.",
            )

        # Read file content
        file_content = await file.read()

        # Save the file temporarily
        temp_file_path = Path(f"temp_{file.filename}")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        # Extract content from the file
        extracted_content = extract_text_from_file(temp_file_path)
        full_content = "\n".join([doc.page_content for doc in extracted_content])

        # Generate questions
        generated_questions = generate_questions(
            full_content, num_questions, question_type
        )

        # Save the generated questions
        output_file = f"generated_quiz.{output_format}"
        output_path = os.path.join(settings.media_dir_static, output_file)
        save_output(generated_questions, output_format, output_path)

        # Remove the temporary file
        os.remove(temp_file_path)
        return make_response(file_path=output_path)
    except Exception as e:
        return make_response({"error": str(e)}, 500)


@api_router.post("/generate_letter_of_recommendation")
async def generate_letter_of_recommendation(
    name: str = "name",
    relationship: str = "relationship",
    skills: str = "skills",
    achievements: str = "achievements",
    purpose: str = "purpose",
    output_format: str = "html",
) -> StreamingResponse:
    try:
        generate_letter = letter_of_recommendation(
            name,
            relationship,
            skills,
            achievements,
            purpose,
        )

        output_file = f"letter_of_recommendation.{output_format}"
        output_path = os.path.join(settings.media_dir_static, output_file)
        save_output(generate_letter, output_format, output_path)
        return make_response(file_path=output_path)
    except Exception as e:
        return make_response({"error": str(e)}, 500)


@api_router.post("/generate_questions_from_YT_videos")
async def generate_questions_from_video(
    youtube_url: str,
    numberOfQuestions: int,
    questionTypes: str,
    gradeLevel: int,
    output_format: str = "html",
) -> StreamingResponse:
    temp_audio_file = download_audio(youtube_url)
    audio_to_text = transcribe_audio(temp_audio_file)
    questions = generate_questions_from_YT_videos(
        numberOfQuestions, questionTypes, gradeLevel, audio_to_text
    )
    output_file = f"questions_from_vid.{output_format}"
    output_path = os.path.join(settings.media_dir_static, output_file)
    save_output(questions, output_format, output_path)
    return make_response(file_path=output_path)
