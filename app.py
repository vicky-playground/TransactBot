from langchain.utilities.twilio import TwilioAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv()


account_sid = "AC0421efcb1131207fb48da1115d291234"
auth_token = "ec3fbf620e3ef6cb7375c5b9767cfed4"
twilio_number = "+12055764467"
to_number = "+14379860835"

twilio = TwilioAPIWrapper(
         account_sid=account_sid,
         auth_token=auth_token,
         from_number=twilio_number
)



twilio.run("hello world", "+14379860835")
