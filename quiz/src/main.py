import time
import threading
from apscheduler.schedulers.background import BackgroundScheduler

from models import QuizLevelEnum
from quiz_generator import generate_level_quiz


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
