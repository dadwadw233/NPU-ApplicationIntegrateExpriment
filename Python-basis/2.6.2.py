
import random
import string

def generate_questions(file_path, n):
    with open(file_path, 'w') as file:
        for i in range(1, n+1):
            question = f"{i}.{generate_random_string(random.randint(1, 50))}\n"
            file.write(question)

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def generate_answer(file_path, n):
    with open(file_path, 'w') as file:
        for i in range(1, n + 1):
            answer = f"{i}.{generate_random_string(random.randint(1, 200))}\n"
            file.write(answer)



def generate_paper(path, questions, answers, n):
    with open(path, 'w') as f:
        indexes = random.sample(range(0, len(questions)), n)
        for i in range(n):
            question = questions[indexes[i]].split('.')[1]
            answer = answers[indexes[i]].split('.')[1]
            content = f"{i+1}.{question}\nanswer: {answer}\n\n"
            f.write(content)


questions_path = './questions.txt'
answers_path = './answer.txt'

generate_questions(questions_path, 1000)
generate_answer(answers_path, 1000)

# process data
questions = []
answers = []

with open(questions_path) as f:
    for line in f:
        questions.append(line.strip())

with open(answers_path) as f:
    for line in f:
        answers.append(line.strip())

generate_paper('./paper.txt', questions, answers, 10)