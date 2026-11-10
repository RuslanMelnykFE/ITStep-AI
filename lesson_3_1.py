import dotenv
import os
import langchain

from langchain_google_genai import GoogleGenerativeAI


dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm = GoogleGenerativeAI(
        model='gemini-3.6-flash',
        api_key=api_key,
    )

# Завдання 1
# Підключіть модель LLM за допомогою свого API key.
# Попросіть модель згенерувати:
# ● відповідь на питання у вигляді одного
# слова(наприклад яка столиця Франції?)
# ● код python
# ● коротку історію
# Підберіть параметри креативності та довжини


def start_task_1(model):

    response = model.invoke("""
        Дай відповідь на питання одним словам
        Яка столиця Бангладешу?
    """)
    print(f"Столиця: \n{response}")

    model.temperature = 0.2
    code = model.invoke("""
        Напиши код phyton виводу суми двох чисел
    """)
    print(f"Код: \n{code}")

    model.temperature = 2.0
    model.top_k = 100
    history = model.invoke("""
        Напиши коротку історію до 5 речень
    """)
    print(f"Історія: \n{history}")


# start_task_1(llm)


# Завдання 2
# Прочитайте файл data\lesson9\rules.txt з правилами
# користування атракціону. Напишіть програму яка отримує
# від користувачі питання та дає відповідь на нього виходячи
# з текстового файлу.
# Для цього об’єднайте правила користування з питанням
# користувача.
# Користувач задає питання поки не введе порожній рядок.
# Змініть файл rules.txt, щоб переконатись що модель
# дійсно його читає.


def start_task_2(model):
    path = "data/lesson9/rules.txt"

    with open(path, 'r', encoding="utf-8") as file:
        rules = file.read()


    while True:
        question = input("Введіть своє питання: ")

        if len(question) == 0:
            break

        response = model.invoke(f"""
            Твоя роль консультант атракціону. 
            Дай коротку та точну відповідь відповідно до правил {rules} на питання {question}.
            Надавай тільки ту інформацію, яка є в правилах, інакше - відповідай не знаєш відповіді.           
        """)

        print(f"Відповідь: \n{response}")


# start_task_2(llm)



# Завдання 3
# Створіть найпростіший чат бот. Напишіть моделі якого
# персонажа вона повинна вдавати(відомий актор, персонаж
# кіно\книги, тощо).
# Реалізуйте двома способами:
# 1. Модель отримує інструкцію в якому стилі відповідати
# та нове повідомлення.
# 2. Модель отримує інструкцію та історію попередніх
# повідомлень як від користувача, так і її власні відповіді у
# форматі
# Instruction: ….
# Human: massage1
# AI: message2
# Human: massage3
# AI: message4
# Human: massage5
# AI: