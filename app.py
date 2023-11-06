import copy
import os
import re

from threading import Thread
from time import sleep
from typing import Dict
from typing import List

from dotenv import load_dotenv
from flask import Flask
from flask import request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from llama_index import SimpleDirectoryReader

# Import necessary libraries
import torch
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, LangchainEmbedding
# Llamaindex also works with langchain framework to implement embeddings 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.prompts.prompts import SimpleInputPrompt
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
llm_hub = None
embeddings = None

Watsonx_API = "uvnQIfnjPk2Jpszy0hAvr80xCUAudclZsltCi3gYxAVu"
Project_id= "177ab670-c7d0-4f34-894f-228297d644d9"

# Function to initialize the Watsonx language model and its embeddings used to represent text data in a form (vectors) that machines can understand. 
def init_llm():
    global llm_hub, embeddings
    
    params = {
        GenParams.MAX_NEW_TOKENS: 250, # The maximum number of tokens that the model can generate in a single run.
        GenParams.MIN_NEW_TOKENS: 1,   # The minimum number of tokens that the model should generate in a single run.
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE, # The method used by the model for decoding/generating new tokens. In this case, it uses the sampling method.
        GenParams.TEMPERATURE: 0.8,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
        GenParams.TOP_K: 50,          # The top K parameter restricts the token generation to the K most likely tokens at each step, which can help to focus the generation and avoid irrelevant tokens.
        GenParams.TOP_P: 1            # The top P parameter, also known as nucleus sampling, restricts the token generation to a subset of tokens that have a cumulative probability of at most P, helping to balance between diversity and quality of the generated text.
    }
    
    credentials = {
        'url': "https://us-south.ml.cloud.ibm.com",
        'apikey' : Watsonx_API
    }

    
    model = Model(
        model_id= 'meta-llama/llama-2-70b-chat',
        credentials=credentials,
        params=params,
        project_id=Project_id)

    llm_hub = WatsonxLLM(model=model)

    #Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )

init_llm()

# load the file
document = SimpleDirectoryReader(input_files=["catalog.txt"]).load_data()

# LLMPredictor: to generate the text response (Completion)
llm_predictor = LLMPredictor(
        llm=llm_hub
)
                                 
# Hugging Face models can be supported by using LangchainEmbedding to convert text to embedding vector	
embed_model = LangchainEmbedding(embeddings)
#embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

# ServiceContext: to encapsulate the resources used to create indexes and run queries    
service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, 
        embed_model=embed_model
)    


# build index
index = GPTVectorStoreIndex.from_documents(document, service_context=service_context)

# use a query engine as the interface for your question
query_engine = index.as_query_engine(service_context=service_context)
# after the query engine is initialized, you can use its `query` method to pass your question as input.
response = query_engine.query("What's the file about?")
print(response)

# Initiate the Flask app
app = Flask(__name__)


def format_activities_text(text: str) -> str:
    """Format the response from the EYFS generator for better display in WhatsApp"""
    text = (
        text.replace("## Conversations", "*Conversations*\n")
        .replace("## Games and Crafts", "*Games and Crafts*\n")
        .replace("**Activity description**", "_Activity description_")
        .replace("**Areas of learning**", "_Areas of learning_")
    )
    # replace markdown subheadings with bold italics
    text = re.sub(r"###\s*(.+)", r"*_\1_*", text)
    return text


def generate_reply(incoming_message: str, sender_contact: str, receiver_contact: str) -> str:
    """Parse message text and return an appropriate response.

    Presently supports two types of responses: 'explain' and 'activities'
    Activities response is threaded to allow for longer response times. This is a very basic
    workaround to the 15 second timeout limit imposed by Twilio.

    Args:
        incoming_message:
            Message text
        sender_contact:
            Sender's contact, follows a format 'whatsapp:+<phone number>'
        receiver_contact:
            Receiver's contact (ie, my contact), follows a format 'whatsapp:+<phone number>'

    Returns:
        Response text
    """
    text_message = incoming_message.lower()

    # 'explain' response
    if text_message[0:7] == "explain":
        response = TextGenerator.generate(
            model=LLM,
            temperature=TEMPERATURE,
            messages=[ELI3_MESSAGES.copy()],
            message_kwargs={"input": text_message[7:].strip()},
        )
        return response["choices"][0]["message"]["content"]
    # 'activities' response
    elif "activities" in text_message[0:10]:
        EYFS_PARAMETERS["description"] = text_message
        thread = Thread(
            target=send_text, args=[copy.deepcopy(EYFS_MESSAGES), EYFS_PARAMETERS, receiver_contact, sender_contact]
        )
        thread.start()
        return "Thank you for your question. I am thinking..."
    else:
        # Return a default message
        return (
            'Write "Explain <your question>" to explain a concept to a 3-year old \n\n or'
            + '\n\n "Activities <your topic>" to get activity ideas'
        )


def send_text(messages: List[Dict], message_kwargs: Dict, my_contact: str, receiver_contact: str) -> None:
    """Generate text messages and send them to a given contact

    Args:
        messages:
            List of messages to be used as prompts
        message_kwargs:
            Dictionary of keyword arguments to be passed to the TextGenerator
        my_contact:
            Sender's contact, follows a format 'whatsapp:+<phone number>'
        receiver_contact:
            Receiver's contact (ie, my contact), follows a format 'whatsapp:+<phone number>'
    """
    # Generate response to the message
    response = query_engine.query(messages)
    text_body = response["choices"][0]["message"]["content"]
    # Format the text_body for better display on WhatsApp
    text_body = format_activities_text(text_body)
    # Divide output into 1500 character chunks due to WhatsApp character limit of 1600 chars
    texts = [text_body[i : i + 1500] for i in range(0, len(text_body), 1500)]
    # Send message
    for text in texts:
        client.messages.create(body=text, from_=my_contact, to=receiver_contact)
        sleep(0.5)
    return


@app.route("/")
def hello_world() -> str:
    """Information message"""
    return "Nesta generative AI prototype: WhatsApp bot for suggesting kids activities"


@app.route("/text", methods=["POST"])
def text_reply() -> str:
    """Respond to incoming messages"""
    reply = generate_reply(
        incoming_message=request.form.get("Body"),
        sender_contact=request.form.get("From"),
        receiver_contact=request.form.get("To"),
    )
    resp = MessagingResponse()
    resp.message(reply)
    return str(resp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
