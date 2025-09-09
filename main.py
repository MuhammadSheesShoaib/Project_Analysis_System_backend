"""
FastAPI Backend for Project Analysis System

This backend system takes a project description and uses Groq's LLM to generate
a complete structured analysis including difficulty, roadmap, tech stack, timeline,
team structure, and cost estimates.

Example Request:
POST /analyze_project
{
    "project_description": "Build an e-commerce platform with AI-powered product recommendations"
}

Example Response:
{
    "difficulty_level": "Hard",
    "project_pipeline": [
        {
            "phase": "Planning & Architecture",
            "duration": "2-3 weeks",
            "deliverables": ["Technical architecture", "Database design", "API specifications"]
        }
    ],
    "tech_stack": {
        "frontend": ["React", "TypeScript", "Tailwind CSS"],
        "backend": ["Node.js", "Express", "Python"],
        "ai_ml": ["TensorFlow", "scikit-learn"],
        "database": ["PostgreSQL", "Redis"],
        "cloud_devops": ["AWS", "Docker", "GitHub Actions"]
    },
    "timeline": {
        "total_duration": "4-6 months",
        "phases": [
            {"phase": "Planning", "duration": "2-3 weeks"},
            {"phase": "Development", "duration": "12-16 weeks"}
        ]
    },
    "team_structure": [
        {
            "role": "Backend Developer",
            "count": 2,
            "time_commitment": "40 hours/week",
            "responsibilities": ["API development", "Database design"]
        }
    ],
    "risks_challenges": [
        "Scalability concerns with high traffic",
        "AI model accuracy and performance"
    ],
    "cost_estimate": "Medium",
    "structured_report": {
        "project_feasibility": "High",
        "complexity_score": 7.5,
        "success_probability": "75%"
    }
}
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import httpx
from groq import Groq
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Project Analysis Backend",
    description="Backend system for analyzing project descriptions using LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ProjectAnalysisRequest(BaseModel):
    """Request model for project analysis"""
    project_description: str = Field(
        ..., 
        min_length=10, 
        max_length=5000,
        description="Detailed description of the project to analyze"
    )
    
    @validator('project_description')
    def validate_description(cls, v):
        if not v.strip():
            raise ValueError('Project description cannot be empty')
        return v.strip()

class PhaseInfo(BaseModel):
    """Model for project phase information"""
    phase: str
    duration: str
    deliverables: List[str]

class TechStack(BaseModel):
    """Model for technology stack"""
    frontend: List[str]
    backend: List[str]
    ai_ml: Optional[List[str]] = []
    database: List[str]
    cloud_devops: List[str]

class TimelinePhase(BaseModel):
    """Model for timeline phase"""
    phase: str
    duration: str

class Timeline(BaseModel):
    """Model for project timeline"""
    total_duration: str
    phases: List[TimelinePhase]

class TeamMember(BaseModel):
    """Model for team member information"""
    role: str
    count: int
    time_commitment: str
    responsibilities: List[str]

class StructuredReport(BaseModel):
    """Model for structured report"""
    project_feasibility: str
    complexity_score: float
    success_probability: str

class ProjectAnalysisResponse(BaseModel):
    """Response model for project analysis"""
    difficulty_level: str
    project_pipeline: List[PhaseInfo]
    tech_stack: TechStack
    timeline: Timeline
    team_structure: List[TeamMember]
    risks_challenges: List[str]
    cost_estimate: str
    structured_report: StructuredReport


load_dotenv()

# Initialize Groq client
def get_groq_client():
    """Initialize and return Groq client"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    return Groq(api_key=groq_api_key)

def create_analysis_prompt(project_description: str) -> str:
    """Create a structured prompt for the LLM to analyze the project"""
    
    prompt = f"""
Analyze the following project description and provide a comprehensive structured analysis in JSON format.

Project Description: {project_description}

Please return ONLY a valid JSON object with the following exact structure (no markdown, no additional text):

{{
    "difficulty_level": "Easy|Medium|Hard|Enterprise",
    "project_pipeline": [
        {{
            "phase": "Phase name",
            "duration": "time estimate",
            "deliverables": ["deliverable1", "deliverable2"]
        }}
    ],
    "tech_stack": {{
        "frontend": ["technology1", "technology2"],
        "backend": ["technology1", "technology2"],
        "ai_ml": ["technology1", "technology2"],
        "database": ["technology1", "technology2"],
        "cloud_devops": ["technology1", "technology2"]
    }},
    "timeline": {{
        "total_duration": "overall project duration",
        "phases": [
            {{
                "phase": "phase name",
                "duration": "phase duration"
            }}
        ]
    }},
    "team_structure": [
        {{
            "role": "role name (e.g., AI Engineer, Backend Dev, Frontend Dev, DevOps, Security, Product Manager)",
            "count": 1,
            "time_commitment": "hours per week or total hours",
            "responsibilities": ["responsibility1", "responsibility2"]
        }}
    ],
    "risks_challenges": ["risk1", "risk2", "risk3"],
    "cost_estimate": "Low|Medium|High",
    "structured_report": {{
        "project_feasibility": "Low|Medium|High",
        "complexity_score": 1.0,
        "success_probability": "percentage"
    }}
}}

Guidelines:
- Be realistic and detailed in your analysis
- Consider modern best practices and technologies
- Include appropriate team roles for the project complexity
- Provide actionable insights
- Ensure all fields are populated with meaningful data
- Complexity score should be between 1.0 and 10.0
"""
    return prompt

async def analyze_with_groq(project_description: str) -> Dict[str, Any]:
    """Analyze project using Groq API"""
    try:
        client = get_groq_client()
        prompt = create_analysis_prompt(project_description)
        
        logger.info(f"Sending request to Groq API for project analysis")
        
        # Make API call to Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert project analyst with deep knowledge of software development, AI/ML, and project management. Provide detailed, realistic, and actionable project analysis in the exact JSON format requested."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=4096,
            top_p=0.9,
            stream=False
        )
        
        # Extract response content
        response_content = chat_completion.choices[0].message.content.strip()
        logger.info(f"Received response from Groq API")
        
        # Clean up response - remove markdown formatting if present
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]
        response_content = response_content.strip()
        
        # Parse JSON response
        try:
            analysis_result = json.loads(response_content)
            logger.info("Successfully parsed JSON response")
            return analysis_result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response_content[:500]}...")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse LLM response. Please try again."
            )
            
    except Exception as e:
        logger.error(f"Error in Groq API call: {str(e)}")
        if "api key" in str(e).lower():
            raise HTTPException(
                status_code=500,
                detail="API configuration error. Please check server settings."
            )
        elif "rate limit" in str(e).lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing project: {str(e)}"
            )

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Project Analysis Backend API",
        "version": "1.0.0",
        "endpoints": {
            "analyze_project": "POST /analyze_project",
            "docs": "GET /docs",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Groq API key availability
        groq_api_key = os.getenv("GROQ_API_KEY")
        api_key_status = "configured" if groq_api_key else "missing"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "groq_api_key": api_key_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.post("/analyze_project", response_model=ProjectAnalysisResponse)
async def analyze_project(request: ProjectAnalysisRequest):
    """
    Analyze a project description using LLM and return structured analysis
    
    This endpoint takes a project description and uses Groq's LLM to generate:
    - Difficulty level assessment
    - Step-by-step project pipeline
    - Recommended technology stack
    - Timeline estimates
    - Team structure and roles
    - Risk assessment
    - Cost estimates
    """
    try:
        logger.info(f"Received project analysis request")
        
        # Analyze project using Groq
        analysis_result = await analyze_with_groq(request.project_description)
        
        # Validate and return response
        try:
            validated_response = ProjectAnalysisResponse(**analysis_result)
            logger.info("Successfully completed project analysis")
            return validated_response
        except Exception as validation_error:
            logger.error(f"Response validation failed: {validation_error}")
            # Log the problematic response for debugging
            logger.error(f"Raw analysis result: {json.dumps(analysis_result, indent=2)}")
            raise HTTPException(
                status_code=500,
                detail="Invalid response format from analysis engine"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_project: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during analysis"
        )

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code} error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable is required!")
        exit(1)
    
    logger.info("Starting Project Analysis Backend API...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )