import os

import pymysql

from models import NewsletterEntity, NewsletterQuizEntity

from dotenv import load_dotenv
load_dotenv()


def get_db_connection():
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        database=os.environ["MYSQL_DATABASE"],   
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
    )


def insert_newsletter_with_quiz(connection, newsletter_entity: NewsletterEntity, newsletter_quiz_entity: NewsletterQuizEntity):
    try:
        with connection.cursor() as cursor:
            newsletter_insert_query = """
            INSERT INTO newsletter (title, category, summary, link, stock, likes, date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            newsletter_quiz_insert_query = """
            INSERT INTO quiz (level, newsletter_id, question, answer, explanation, wrong_answer1, wrong_answer2, wrong_answer3)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            newsletter_values = (
                newsletter_entity.title,
                newsletter_entity.category,
                newsletter_entity.summary,
                newsletter_entity.link,
                newsletter_entity.stock,
                newsletter_entity.likes,
                newsletter_entity.date
            )
            cursor.execute(newsletter_insert_query, newsletter_values)
            newsletter_id = cursor.lastrowid

            newsletter_quiz_values = (
                newsletter_quiz_entity.level,
                newsletter_id,
                newsletter_quiz_entity.question,
                newsletter_quiz_entity.answer,
                newsletter_quiz_entity.explanation,
                newsletter_quiz_entity.wrong_answer1,
                newsletter_quiz_entity.wrong_answer2,
                newsletter_quiz_entity.wrong_answer3
            )
            cursor.execute(newsletter_quiz_insert_query, newsletter_quiz_values)
            
            connection.commit()
            print("뉴스레터 데이터가 성공적으로 저장되었습니다.")

    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
        raise e


def clear_all_newsletters():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            delete_query = "DELETE FROM newsletter"
            cursor.execute(delete_query)

            delete_query = "DELETE FROM quiz where level = 3"
            cursor.execute(delete_query)

            connection.commit()
            print("newsletter 테이블 및 quiz 테이블에 있는 모든 뉴스레터 퀴즈가 삭제되었습니다.")
    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
    finally:
        connection.close()


def get_all_newsletters():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            select_query = """
            SELECT title, category, summary, link, stock, likes, date 
            FROM newsletter
            ORDER BY date DESC
            """
            cursor.execute(select_query)
            results = cursor.fetchall()
            
            print("\n=== 뉴스레터 목록 ===")
            for row in results:
                print(f"\n제목: {row[0]}")
                print(f"카테고리: {row[1]}")
                print(f"요약: {row[2]}")
                print(f"링크: {row[3]}")
                print(f"주식: {row[4]}")
                print(f"좋아요 수: {row[5]}")
                print(f"날짜: {row[6]}")
                print("-" * 50)
            
            print(f"\n총 {len(results)}개의 뉴스레터가 있습니다.")
            
    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
    finally:
        connection.close()