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
# Define prompt templates
prompt = PromptTemplate(
    input_variables=["name", "relationship", "skills", "achievements", "purpose"],
    template=(
        """
        Write a professional letter of recommendation in the format of a CV with the following details:

        Name of the person being recommended: {name}
        Your relationship to them: {relationship}
        Key skills or qualities: {skills}
        Achievements or examples: {achievements}
        Purpose of the recommendation: {purpose}

        Format it with the following structure:

        --------------------------------------------------------------
        [Name of the Person Being Recommended]
        [Purpose of Recommendation, e.g., "Candidate for MBA Program"]
        --------------------------------------------------------------

        PROFESSIONAL RELATIONSHIP
        [Describe your relationship with them, e.g., "I supervised them during their tenure as a software engineer at BrightTech Solutions."]

        SKILLS
        - [List skills, e.g., "Leadership and communication"]
        - [Skill]
        - [Skill]

        ACHIEVEMENTS
        - [Example 1, e.g., "Led a team that delivered a critical software project ahead of schedule."]
        - [Example 2]
        - [Example 3]

        PURPOSE OF RECOMMENDATION
        [Explain why you recommend this individual for their specific purpose, e.g., "Emily Johnson is an exceptional candidate for the MBA program."]

        CONTACT INFORMATION
        [Recommender's Name]
        [Title]
        [Organization]
        [Email | Phone Number]
        --------------------------------------------------------------
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
