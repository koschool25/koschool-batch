import time
import threading

from tqdm import tqdm
from apscheduler.schedulers.background import BackgroundScheduler

from models import NewsletterCategoryEnum
from newsletter_extractor import extract_category_newsletters


def main():
    threads = []
    for category in tqdm(NewsletterCategoryEnum):
        print(f"{category.value} 뉴스 저장 시작...")
        thread = threading.Thread(target=extract_category_newsletters, args=(category.value,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=main,
        trigger='cron',
        hour=6,
        minute=0,
        second=0,
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
