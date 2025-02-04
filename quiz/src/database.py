import os
import pymysql

from dotenv import load_dotenv
from models import QuizEntity
load_dotenv()


def get_db_connection():
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        database=os.environ["MYSQL_DATABASE"],   
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
    )


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
            delete_query = "DELETE FROM quiz"
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
                print(f"\nID: {row[0]}")
                print(f"레벨: {level_int}")
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