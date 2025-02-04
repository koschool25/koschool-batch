from enum import Enum
from pydantic import BaseModel


class QuizGenerationError(Exception):
    pass


class QuizLevelEnum(Enum):
    BEGINNER = "하수"
    INTERMEDIATE = "중수"
    EXPERT = "고수"
    
    @classmethod
    def to_int(cls, value: str) -> int:
        int_mapping = {
            "하수": 0,
            "중수": 1,
            "고수": 2,
        }
        return int_mapping.get(value, -1)


class QuizEntity(BaseModel):
    level: int
    question: str
    answer: str
    explanation: str
    wrong_answer1: str
    wrong_answer2: str
    wrong_answer3: str


class GeneratedQuiz(BaseModel):
    question: str
    answer: str
    explanation: str
    multipleChoices: list[str]
    keyword: str

    def to_quiz_entity(self, level: QuizLevelEnum) -> QuizEntity:
        return QuizEntity(
            level=QuizLevelEnum.to_int(level),
            question=self.question,
            answer=self.answer,
            explanation=self.explanation,
            wrong_answer1=self.multipleChoices[0],
            wrong_answer2=self.multipleChoices[1],
            wrong_answer3=self.multipleChoices[2],
        )
