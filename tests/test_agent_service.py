from lc_agent.core.agent import AgentService


def test_extract_usage_reads_last_model_and_usage() -> None:
    from langchain_core.messages import AIMessage

    messages = [
        AIMessage(
            content="hello",
            usage_metadata={
                "input_tokens": 3,
                "output_tokens": 4,
                "total_tokens": 7,
            },
            response_metadata={"model": "fake-model"},
        )
    ]

    usage = AgentService.extract_usage(messages)

    assert usage.input_tokens == 3
    assert usage.output_tokens == 4
    assert usage.total_tokens == 7
    assert usage.model == "fake-model"
