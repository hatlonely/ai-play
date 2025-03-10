from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import OllamaLLM

from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate


def search_and_summarize(query):
    # 初始化搜索工具，设置超时和重试
    search = DuckDuckGoSearchRun(
        timeout=10,  # 10秒超时
        max_retries=3  # 最多重试3次
    )

    # 初始化 Ollama 模型
    llm = OllamaLLM(model="qwq:32b")

    # 定义工具
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="用于搜索互联网最新信息"
        )
    ]

    # 初始化 agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    # 定义提示模板
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        请搜索以下主题的最新信息并总结：
        {query}
        
        要求：
        1. 使用互联网搜索获取最新信息
        2. 总结内容要简洁明了
        3. 包含关键事实和数据
        """
    )

    try:
        # 执行搜索和总结
        summary = agent.run(prompt.format(query=query))
        return summary
    except Exception as e:
        # 如果搜索失败，返回友好的错误信息
        error_msg = f"搜索失败：{str(e)}"
        print(error_msg)
        return f"无法获取最新信息。请稍后重试。\n错误详情：{error_msg}"


if __name__ == "__main__":
    query = "2025年最新的人工智能发展趋势"
    result = search_and_summarize(query)
    print("总结结果：")
    print(result)
