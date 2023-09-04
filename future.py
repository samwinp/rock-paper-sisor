import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair

vertexai.init(project="samwin", location="us-central1")
chat_model = ChatModel.from_pretrained("chat-bison@001")
parameters = {
    "max_output_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
chat = chat_model.start_chat(
    context="""you are a doctor that is responsible for answering queries that patients regarding their prescription. Also you are not allowed to tell that you are an AI model and have to answer the question with full confidence""",
    examples=[
        InputOutputTextPair(
            input_text="""hello """,
            output_text="""hey there how can I help you out """
        ),
        InputOutputTextPair(
            input_text="""can you tell me your name """,
            output_text="""My name is doctor Charis your person medical assistant """
        ),
        InputOutputTextPair(
            input_text="""is it dangerous to take Asparin  during Dengu""",
            output_text="""yes it is dangerous to take Asparin as it is a blood thinner and can cause excess thinning """
        )
    ]
)

data = input("enter the text: ")
response = chat.send_message(f'{data}' , **parameters)
print(f"Response from the model {response.text}")