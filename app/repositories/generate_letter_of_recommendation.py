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

# Define prompt templates
prompt = PromptTemplate(
    input_variables=["name", "relationship", "skills", "achievements", "purpose"],
    template=(
        """
        Write a professional letter of recommendation based on the following details:

        Name of the person being recommended: {name}
        Your relationship to them: {relationship}
        Key skills or qualities: {skills}
        Achievements or examples: {achievements}
        Purpose of the recommendation: {purpose}

        Make it formal, personalized, and compelling.
        """
    ),
)


def letter_of_recommendation(name, relationship, skills, achievements, purpose):
    chain = prompt | llm
    response = chain.invoke(
        {
            "name": name,
            "relationship": relationship,
            "skills": skills,
            "achievements": achievements,
            "purpose": purpose,
        }
    )
    return response.content if hasattr(response, "content") else str(response)
