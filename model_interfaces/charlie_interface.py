import json
from dataclasses import dataclass, field
from typing import List

import browser_cookie3
from requests import Session

from model_interfaces.interface import ChatSession
import requests

from utils.constants import ResetPolicy


@dataclass
class CharlieMnemonic(ChatSession):
    context: List[str] = field(default_factory=list)
    max_prompt_size: int = 8192
    chat_id: str = "Benchmark"
    url: str = "127.0.0.1"
    port: str = "8002"
    token: str = ""
    user_name: str = "admin"
    password: str = "admin"
    initial_costs_usd: float = 0.0
    session: Session = field(default_factory=requests.Session)
    initialised: bool = False

    @property
    def name(self):
        return f"{super().name} - {self.max_prompt_size}"

    @property
    def endpoint(self):
        return "http://" + self.url + ":" + self.port

    def __post_init__(self):
        super().__post_init__()
        self.login()

        # Get display name and current costs of user
        settings_dict = self.get_settings()
        self.display_name = settings_dict["display_name"]
        self.initial_costs_usd = settings_dict["usage"]["total_cost"]

        #TODO: Setting of max tokens


    def login(self):
        body = {
            "username": self.user_name,
            "password": self.password,
        }

        response = self.session.post(self.endpoint + "/login/", json=body)

        if response.status_code == 200:
            print("Login successful.")
            # Extract the session token and username from the response cookies
            session_token = response.cookies.get("session_token")
            username = response.cookies.get("username")
            # Set the session token and username cookies in the session object
            self.session.cookies.set("session_token", session_token)
            self.session.cookies.set("username", username)
        else:
            raise ValueError(f"Login failed Status code: {response.status_code}, Response: {response.text}")

    def reply(self, user_message) -> str:
        if not self.initialised:
            self.reset()

        message_data = {
            "prompt": user_message,
            "username": self.user_name,
            "display_name": self.display_name,
            "chat_id": self.chat_id,
        }

        response = self.session.post(self.endpoint + "/message/", json=message_data)

        if response.status_code == 200:
            # Update costs
            settings = self.get_settings()
            self.costs_usd = settings["usage"]["total_cost"] - self.initial_costs_usd

            return response.text
        else:
            raise ValueError(f"Failed to send message. Status code; {response.status_code}, Response: {response.text}")

    def get_settings(self):
        body = {"username": self.user_name}
        settings = self.session.post(self.endpoint + "/load_settings/", json=body)
        return json.loads(settings.text)

    def reset(self):
        self.initialised = True

        # Delete the user and memory data
        delete_req = self.session.post(self.endpoint + "/delete_data_keep_settings/")
        #
        # # Erase the context/chat
        # body = {"username": self.user_name, "chat_id": self.chat_id}
        # chat_delete_req = self.session.post(self.endpoint + "/delete_chat_tab/", json=body)
        a = 1

    def load(self):
        # Charlie mnemonic is web based and so doesn't need to be manually told to resume a conversation
        self.initialised = True
        #TODO: We might need to get the correct chat

    def save(self):
        # Charlie mnemonic is web based and so doesn't need to be manually told to persist
        pass
