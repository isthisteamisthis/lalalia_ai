import locale
import json
import os, sys

model_dir = os.getcwd() + "\\pyaicover\\models"
sys.path.append(model_dir)

def load_language_list(language):
    with open(f"{model_dir}/i18n/{language}.json", "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        if not os.path.exists(f"./pyaicover/models/i18n/{language}.json"):
            language = "en_US"
        self.language = language
        # print("Use Language:", language)
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def print(self):
        print("Use Language:", self.language)
