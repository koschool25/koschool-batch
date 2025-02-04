import random

from typing import Optional
from enum import Enum
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field


class NewsletterExtractionError(Exception):
    pass


class NewsletterCategoryEnum(Enum):
    ENERGY = "에너지"
    MATERIALS = "소재"
    INDUSTRIALS = "산업"
    CONSUMER_DISCRETIONARY = "임의소비재"
    CONSUMER_STAPLES = "필수소비재"
    HEALTHCARE = "헬스케어"
    FINANCIALS = "금융"
    INFORMATION_TECHNOLOGY = "정보기술"
    COMMUNICATION = "통신"
    UTILITIES = "유틸리티"
    REAL_ESTATE = "부동산"


class SearchedNewsletter(BaseModel):
    title: str
    link: str
    publish_date: datetime


class NewsletterQuiz(BaseModel):
    question: str
    answer: str
    explanation: str
    wrong_answer1: str
    wrong_answer2: str
    wrong_answer3: str


class NewsletterQuizEntity(BaseModel):
    level: int
    question: str
    answer: str
    explanation: str
    wrong_answer1: str
    wrong_answer2: str
    wrong_answer3: str


class ExtractedNewsletter(BaseModel):
    summary: str
    relative_stock: Optional[str] = None
    quiz: NewsletterQuiz

    def to_newsletter_quiz_entity(self) -> NewsletterQuizEntity:
        return NewsletterQuizEntity(
            level=3,
            question=self.quiz.question,
            answer=self.quiz.answer,
            explanation=self.quiz.explanation,
            wrong_answer1=self.quiz.wrong_answer1,
            wrong_answer2=self.quiz.wrong_answer2,
            wrong_answer3=self.quiz.wrong_answer3,
        )


class NewsletterEntity(BaseModel):
    title: str
    category: str
    summary: str
    link: str
    stock: str
    likes: int = Field(default_factory=lambda: random.randint(10, 200))
    date: datetime = Field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Seoul")).date())
    
    @classmethod
    def make_newsletter(
        cls, 
        category: NewsletterCategoryEnum,
        searched_newsletter: SearchedNewsletter, 
        extracted_newsletter: ExtractedNewsletter
    ):
        return cls(
            title=searched_newsletter.title,
            category=category,
            summary=extracted_newsletter.summary,
            link=searched_newsletter.link,
            stock=extracted_newsletter.relative_stock,
        )