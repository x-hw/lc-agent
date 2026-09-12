from lc_agent.core.agent import make_agent


def main():
    agent = make_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "列出当前工作目录的文件"}]}
    )
    print(result["messages"])


if __name__ == "__main__":
    main()
