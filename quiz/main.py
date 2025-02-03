import os
import time
import json
import openai
import threading

import pymysql

from enum import Enum
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler

from dotenv import load_dotenv
load_dotenv()


def get_db_connection():
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        database=os.environ["MYSQL_DATABASE"],   
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
    )


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


class QuizGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    

    def generate(self, level: QuizLevelEnum, previous_keywords: list[str]) -> GeneratedQuiz:
        system_prompt = f"""당신은 경제와 투자 분야의 퀴즈 생성 전문가입니다. 주어진 레벨({level})에 맞는 퀴즈를 생성해주되, 다음 키워드들과 중복되지 않는 새로운 주제의 퀴즈를 생성해주세요.
---
이전 출제된 키워드 목록:
{previous_keywords}
위 키워드들과 관련된 주제는 제외하고 출제해주세요.
---
레벨별 출제 기준:
- {QuizLevelEnum.BEGINNER.value} : 경제/투자에 대한 기초적인 지식도 부족한 중학생 대상
- {QuizLevelEnum.INTERMEDIATE.value} : 기초적인 경제/투자 용어와 개념, 일상생활에서 자주 접하는 금융 용어는 아는 고등학생 대상
- {QuizLevelEnum.EXPERT.value} : 비금융권 회사에 재직중이지만 금융 관련 자격증이 있는 전문가 대상
---
JSON 각 필드별 요구사항:
- level: 문제 출제 레벨
- question: 지정된 레벨에 맞는 명확하고 간단한 주관식 질문
- answer: 정확한 정답 (주관식 답안으로 사용)
- explanation: 해당 레벨 수준에 맞는 상세한 설명과 관련 맥락
- multipleChoices: 4개의 선택지 배열 (첫 번째 요소가 반드시 정답)
- keyword: 퀴즈 핵심 키워드 (이전 키워드와 중복되지 않아야 함)
---
중복 방지 규칙:
1. 이전 키워드와 동일한 주제는 출제하지 않음
2. 유사한 개념이더라도 다른 관점이나 심화된 내용이라면 출제 가능
3. keyword는 반드시 이전 목록에 없는 새로운 것이어야 함
---
퀴즈 작성 지침:
1. 한국어를 사용
2. 주로 주식 및 투자와 관련된 퀴즈를 출제
3. 지정된 레벨에 맞는 용어와 개념을 사용
4. 주관식 답을 맞출 수 있도록 되도록 정답은 단어 및 용어 위주로 출제
5. 문제에 정답이 포함되지 않도록 출제
6. 객관식 선택지는 해당 레벨에 적절한 수준으로 구성
7. 설명은 해당 난이도의 학습자가 이해할 수 있는 수준으로 작성
8. JSON 형식이 올바르게 유지되어야 함
---
지정된 레벨 {level}에 맞고, 이전 키워드와 중복되지 않는 새로운 퀴즈를 한국어로 생성해주세요.
"""

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            temperature=1,
            response_format=GeneratedQuiz,
        )

        result = completion.choices[0].message
        if result.refusal:
            raise QuizGenerationError(f"퀴즈 생성이 거부되었습니다: {result.refusal}")
        else:
            generated_quiz = result.parsed

            return generated_quiz


def generate_level_quiz(level: QuizLevelEnum):
    try:
        connection = get_db_connection()
        openai_api_key = os.environ["OPENAI_API_KEY"]

        quiz_generator = QuizGenerator(openai_api_key)
        previous_keywords = []
        n_quiz = int(os.getenv("N_QUIZ", 100))
        for i in tqdm(range(n_quiz)):
            MAX_RETRIES = 3
            retry_count = 0
            
            while retry_count < MAX_RETRIES:
                try:
                    generated_quiz = quiz_generator.generate(level, previous_keywords)
                    quiz_entity = generated_quiz.to_quiz_entity(level)
                    insert_quiz(connection, quiz_entity)
                    previous_keywords.append(generated_quiz.keyword)
                    break
                except QuizGenerationError as e:
                    retry_count += 1
                    if retry_count >= MAX_RETRIES:
                        print(f"최대 재시도 횟수({MAX_RETRIES})를 초과했습니다: {str(e)}")
                        return
                    print(f"퀴즈 생성 실패 ({retry_count}/{MAX_RETRIES}), 재시도 중...")
                except openai.RateLimitError as e:
                    print("OpenAI API Rate limit 도달... 60초 후에 작업 이어서 진행")
                    time.sleep(60)
    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
    finally:
        connection.close()


def insert_quiz(connection, quiz_entity: QuizEntity):
    try:
        with connection.cursor() as cursor:
            insert_query = """
            INSERT INTO quiz (level, question, answer, explanation, wrong_answer1, wrong_answer2, wrong_answer3)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            values = (
                quiz_entity.level,
                quiz_entity.question,
                quiz_entity.answer,
                quiz_entity.explanation,
                quiz_entity.wrong_answer1,
                quiz_entity.wrong_answer2,
                quiz_entity.wrong_answer3
            )
            
            cursor.execute(insert_query, values)
            connection.commit()

    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
        raise e


def clear_quiz_table():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            delete_query = "DELETE FROM quiz WHERE level = 3"
            cursor.execute(delete_query)
            connection.commit()
            print("quiz 테이블의 모든 데이터가 삭제되었습니다.")
    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
    finally:
        connection.close()


def get_all_quizzes():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            select_query = """
            SELECT id, level, question, answer, explanation, wrong_answer1, wrong_answer2, wrong_answer3 
            FROM quiz
            ORDER BY level, id
            """
            cursor.execute(select_query)
            results = cursor.fetchall()
            
            print("\n=== 퀴즈 목록 ===")
            for row in results:
                level_int = row[1]
                level_str = "하수" if level_int == 0 else "중수" if level_int == 1 else "고수"
                print(f"\nID: {row[0]}")
                print(f"레벨: {level_str}")
                print(f"문제: {row[2]}")
                print(f"정답: {row[3]}")
                print(f"설명: {row[4]}")
                print(f"오답1: {row[5]}")
                print(f"오답2: {row[6]}")
                print(f"오답3: {row[7]}")
                print("-" * 50)
            
            print(f"\n총 {len(results)}개의 퀴즈가 있습니다.")
            
    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
    finally:
        connection.close()


def main():
    # 하수, 중수, 고수 레벨 퀴즈 병렬적으로 생성
    threads = []
    for level in QuizLevelEnum:
        print(f"{level.value} 퀴즈 생성 시작...")
        thread = threading.Thread(target=generate_level_quiz, args=(level.value,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=main,
        trigger='cron',
        day_of_week=0,
        args=[]
    )
    
    try:
        scheduler.start()
        print("스케줄러 실행...")
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("스케줄러 종료...")
