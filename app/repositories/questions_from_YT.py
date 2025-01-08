import yt_dlp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from app.core.settings import settings

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=settings.gemini_key,
    temperature=0.5,
    max_tokens=1024,
    timeout=30,
    max_retries=3,
)


def download_audio(youtube_url):
    """Download audio from a YouTube video using yt-dlp."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "audio.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        return f"audio.{ydl_opts['postprocessors'][0]['preferredcodec']}"


def transcribe_audio(audio_file):
    """Transcribe audio to text using Gemini"""
    prompt = PromptTemplate("Transcribe is mp3 file to text {file}")
    with open(audio_file, "rb") as file:
        chain = prompt | llm
        transcribed_audio = chain.invoke({"file": file})
    return transcribed_audio


def generate_questions_from_YT_videos(
    transcription, numberOfQuestions, questionTypes, gradeLevel
):
    """Generate questions using the Gemini API"""
    prompt = PromptTemplate(
        template=(
            "Based on the following video transcript, generate {numberOfQuestions} "
            "{questionTypes} questions suitable for grade {gradeLevel}: \n\n{transcription}"
        )
    )
    transcription = transcribe_audio()
    chain = prompt | llm
    response = chain.invoke(
        {
            "numberOfQuestions": numberOfQuestions,
            "questionTypes": {questionTypes},
            "gradeLevel": {gradeLevel},
            "transcription": {transcription},
        }
    )
    return response.content if hasattr(response, "content") else str(response)
