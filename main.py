import requests
from bs4 import BeautifulSoup
import telegram
import asyncio
import os, time
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline


# 연합뉴스 (오늘)
def get_news_from_yonhap(): 
    # URL of the RSS feed
    url = "https://www.yonhapnewstv.co.kr/category/news/economy/feed/"

    # Fetch the RSS feed
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses

    # Parse the XML
    soup = BeautifulSoup(response.text, "xml")
    # 뉴스 제목
    titles = [item.find('title').text for item in soup.find_all('item')]
    # 뉴스 내용
    encoded_contents = [item.find('content:encoded').text for item in soup.find_all('item')]
    # 링크
    links = [item.find('link').text for item in soup.find_all('item')]
    return titles, encoded_contents, links


# 호악재 분석 (간단 키워드 버전)
def analyze_news_simple(news):
    호재_키워드 = ['성장', '상승', '증가', '이익', '호조', '발전', '호재', '성과', '성공', '성장세', '성장률', '육성', '증대', '최고치']
    악재_키워드 = ['감소', '하락', '손실', '악화', '위기', '악재', '부진', '위축', '위험', '위기', '추락', '하락세', '하락률', '우려', '부정적', '부정', '부정적인', '장애']

    호재_점수 = sum(키워드 in news for 키워드 in 호재_키워드)
    악재_점수 = sum(키워드 in news for 키워드 in 악재_키워드)

    if 호재_점수 > 악재_점수:
        return "호재"
    elif 악재_점수 > 호재_점수:
        return "악재"
    else:
        return "중립"


# 호악재 분석 (고급 버전) : Hugging Face 모델 사용
def analyze_news(pipe, text):    
    # 한번에 분석하기 힘든 긴 텍스트 때문에 청크로 자른다. 
    max_length = 512  # Assuming the model's max token length is 512
    text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

    # 예측을 진행하고 결과를 청크별로 나눠 담는다
    results = []
    for chunk in text_chunks:
        result = pipe(chunk)
        result[0]['length'] = len(chunk)
        results.append(result)

    # 청크별 글자수가 다르므로 가중치를 부여한다
    weighted_results = {}
    for result in results:
        for res in result:
            label = res['label']
            length = res['length']
            if label in weighted_results:
                weighted_results[label] += length
            else:
                weighted_results[label] = length

    # 가장 빈도가 높은 결과를 최종 결과로 도출한다.
    most_common_label = max(weighted_results, key=weighted_results.get)
    mapping = {"positive": "긍정", "negative": "부정", "neutral": "중립"}
    
    return mapping[most_common_label]


# 메시지 전송 : 텔레그램
async def sendMessage(msg):
    TGM_TOKEN = os.environ.get('TGM_TOKEN')
    TGM_CHAT_ID = os.environ.get('TGM_CHAT_ID')
    bot = telegram.Bot(token=TGM_TOKEN)
    await bot.sendMessage(chat_id=TGM_CHAT_ID, text=msg)


# 메시지 전송 호출 함수 
def message(msg):
    asyncio.run(sendMessage(msg))


# 메인 실행 함수 
if __name__ == "__main__":
    start_time = time.time()  # 시작 시간 측정
    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained("gabrielyang/finance_news_classifier-KR_v7")
    tokenizer = AutoTokenizer.from_pretrained("gabrielyang/finance_news_classifier-KR_v7")
    # pipeline 시작
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # 뉴스 제목, 내용 가져오기
    title, news, links = get_news_from_yonhap()
    # 뉴스 내용 분석 
    results = []
    for idx, contents in enumerate(news):
        # simple version
        # predict = analyze_news_simple(i)        
        # advanced version
        predict = analyze_news(pipe, contents)

        # 결과 문자열 만들기 : 뉴스제목 / 분석결과 / 점수 / 원문링크        
        result = "%s \n > 분석결과 : %s (원문 : %s)" % (title[idx], predict, links[idx])

        results.append(result)
    # 메시지 전송
    results_string = '\n'.join(results)
    message(results_string)
    end_time = time.time()  # 종료 시간 측정
    print("실행 시간 : ", end_time - start_time, "뉴스 건수 : ", len(news))  # 수행 시간 출력
    print("메시지 전송 완료")