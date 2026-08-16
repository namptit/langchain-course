from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()
def main():
    print("Hello from langchain-course!")
    information = """
        Phạm Nhật Vượng (sinh ngày 5 tháng 8 năm 1968) là một doanh nhân kiêm tỷ phú người Việt Nam. Ông là nhà sáng lập và Chủ tịch Hội đồng quản trị của Tập đoàn Vingroup, tập đoàn tư nhân đa ngành có vốn hóa lớn nhất Việt Nam.[1]

Được ghi nhận là tỷ phú đô la Mỹ đầu tiên trên sàn chứng khoán Việt Nam,[3] ông hiện là người giàu nhất Việt Nam và cũng là người giàu nhất Đông Nam Á tính đến tháng 4 năm 2026, với giá trị tài sản ròng ước tính khoảng 22,1 tỷ USD theo Forbes
    """
    
    summary_template = """
        given the information {information} about a person, create:
        1. A short summary
        2. two intesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
