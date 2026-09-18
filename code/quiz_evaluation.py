import os
import json
import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from statistics import mean

# Load environment variables
load_dotenv()

# Initializing the model
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key = api_key
)

model_persona = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key = api_key
)

# Data Structures
class CriterionScore(BaseModel):
    score: int
    justification: str
 
class QuestionEvaluation(BaseModel):
    topic_matching: CriterionScore
    topic_grounding: CriterionScore
    answer_exclusivity: CriterionScore
    distractor_quality: CriterionScore
    hint_quality: CriterionScore
    rationale_quality: CriterionScore

class StudentAnswer(BaseModel):
    chosen_text: str
    chosen_option: str  # A, B, C, D
    reasoning: str

# Evaluate Quiz Question
# LLM as a judge: Topic Grounding, Answer Exclusivity, Hint Quality, Rationale Quality
def evaluate_question_topic(topics, question, tag, source_content_tag):

    SYSTEM_PROMPT = """Vais avaliar uma pergunta de um quiz educativo de literacia financeira, gerado a partir de artigos-fonte (RAG), segundo seis critérios.
    Para cada critério, atribui uma classificação de 1 a 5 e justifica em 1-2 frases.
    
    1. Adequação ao Tópico e Tags (topic_matching)

    Avalia apenas se a pergunta está corretamente associada ao tópico/tag atribuído.
    
    Escala:
    1 - A pergunta não é relevante para o tópico/tag atribuído. O conteúdo pertence claramente a outro tema.
    2 - A pergunta tem uma ligação mínima com o tópico/tag, mas o enquadramento é fraco ou pouco representativo.
    3 - A pergunta está relacionada com o tópico geral, mas o alinhamento com a tag específica é parcial ou poderia ser melhor classificada.
    4 - A pergunta está bem alinhada com o tópico/tag e avalia conhecimento relevante desse domínio.
    5 - A pergunta está diretamente relacionada com o tópico/tag atribuído e representa claramente o conhecimento que esse tópico pretende avaliar.

    2. Fundamentação no Tópico (topic_grounding)

    Avalia se a pergunta e a resposta correta estão corretamente sustentadas pelo conteúdo-fonte.

    Escala:
    1 – A pergunta e a resposta correta contêm informação claramente incorreta ou que vão contra o conteúdo-fonte.
    2 – Existem problemas relevantes de fundamentação e correção na pergunta e resposta correta, tendo em conta o conteúdo-fonte.
    3 – A pergunta e a resposta correta apresentam fundamentação parcial, tendo em conta o conteúdo-fonte.
    4 – A pergunta e a resposta correta estão bem fundamentadas pelo conteúdo-fonte, existindo apenas pequenas imprecisões.
    5 – A pergunta e a resposta correta estão totalmente consistentes e bem fundamentadas pelo conteúdo-fonte.
    
    3. Exclusividade das Respostas (answer_exclusivity)

    Avalia se as opções de resposta são claras, mutuamente exclusivas e inequívocas, existindo apenas uma resposta correta.

    Escala:
    1 – Mais do que uma opção pode ser considerada correta ou a resposta correta é ambígua.
    2 – Existe ambiguidade relevante entre pelo menos duas opções.
    3 – Existe alguma ambiguidade, mas uma opção destaca-se como a mais correta.
    4 – As opções são claras e distintas, com apenas pequenas imperfeições.
    5 – As opções são totalmente claras e existe apenas uma resposta correta, sem ambiguidades.
    
    4. Qualidade dos Distratores (distractor_quality)

    Avalia se as opções incorretas são alternativas plausíveis, relacionadas com o conceito avaliado e construídas de forma adequada.

    Escala:
    1 - Os distratores são incorretos de forma evidente, absurdos, irrelevantes ou não relacionados com o conceito avaliado.
    2 - Os distratores têm alguma relação com o tema, mas apresentam erros claros, formulações inadequadas ou pouca plausibilidade.
    3 - Os distratores são relacionados com o tema, mas alguns apresentam diferenças conceptuais óbvias relativamente à resposta correta.
    4 - Os distratores são plausíveis e relacionados com o conceito avaliado, embora possam existir pequenas diferenças de qualidade entre opções.
    5 - Os distratores são plausíveis, conceptualmente coerentes e representam possíveis erros ou interpretações incorretas que um participante poderia efetivamente ter.

    5. Qualidade da Dica (hint_quality)

    Avalia se a dica ajuda o participante a raciocinar sobre a resposta sem revelar diretamente a solução.

    Escala:
    1 – A dica revela diretamente a resposta ou não fornece qualquer ajuda útil.
    2 – A dica revela informação em excesso ou é praticamente inútil.
    3 – A dica fornece alguma orientação, mas de forma genérica ou limitada.
    4 – A dica orienta o raciocínio de forma útil, mas de forma incompleta ou pouco direcionada.
    5 – A dica orienta eficazmente o raciocínio, sem fornecer pistas excessivas nem revelar a resposta.
    
    6. Qualidade do Racional (rationale_quality)

    Avalia se a explicação apresentada é correta, clara, completa e consistente com o conteúdo-fonte, justificando adequadamente a resposta correta.

    Escala:
    1 – A explicação está incorreta, é circular (por exemplo, "é A porque é A") ou é incompreensível.
    2 – A explicação está parcialmente correta, mas é superficial ou contém erros relevantes.
    3 – A explicação está correta, mas é incompleta ou pouco clara.
    4 – A explicação está correta, clara e praticamente completa, com pequenas possibilidades de melhoria.
    5 – A explicação está correta, completa, clara e estabelece explicitamente a ligação ao conteúdo-fonte.
    """

    # Send user request
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            SYSTEM_PROMPT
        ),
        (
            "user",
            """
                Avalia esta pergunta de acordo com a informação fornecida:

                Tópicos:
                {topics}

                Tag:
                {tag}

                Conteúdo-fonte:
                {source_content_tag}

                Pergunta a avaliar:
                {question}

            """
        )
    ])

    structured_model = model.with_structured_output(QuestionEvaluation)

    chain = prompt | structured_model

    response = chain.invoke({
        "topics": topics,
        "tag": tag,
        "source_content_tag": source_content_tag if source_content_tag else "Nenhum conteúdo-fonte encontrado.",
        "question": json.dumps(question, ensure_ascii = False, indent = 2)
    })

    return response

# Personas for quiz difficulty
# Benedetto et al., 2024, "Using LLMs to simulate students' responses to exam questions"

# Persona prompts were deliberately calibrated to be more conservative than their nominal literacy level. Preliminary experiments showed that using literal descriptions (e.g., "average financial literacy") resulted in unrealistically high performance due to the language model's inherent financial knowledge. Therefore, persona descriptions were shifted downward by approximately one literacy level to better approximate human response distributions.
PERSONAS = {
    "Baixo": """És uma pessoa de Portugal com literacia financeira MUITO BAIXA.
    - Não conheces termos técnicos de finanças ou siglas (ex: TAEG, spread, capitalização composta, diversificação).
    - Se a pergunta usar jargão financeiro, é natural que interpretes mal ou fiques confuso.
    - É esperado e aceitável que erres perguntas técnicas.
    - Respondes sempre com base em senso comum do dia-a-dia, nunca com cálculos ou definições técnicas.
    - Respondes como realmente responderia uma pessoa assim, mesmo que a resposta esteja errada.
    """,

    "Médio": """És uma pessoa de Portugal com literacia financeira BAIXA.
    - Conheces conceitos básicos de orçamento, poupança e crédito (ex: juros simples, cartão de crédito, prestação mensal).
    - Se a pergunta usar conceitos mais avançados ou siglas desconhecidas, é natural que interpretes mal ou fiques confuso.
    - Não tens memória fiável de percentagens, limites legais, prazos ou valores concretos.
    - É esperado e aceitável que erres perguntas sobre conceitos avançados.
    - Respondes usando linguagem do dia-a-dia e raciocínio básico, em vez de explicações técnicas ou demasiado rigorosas.
    - Respondes como realmente responderia uma pessoa assim, mesmo que a resposta esteja errada.
    """,

    "Alto": """És uma pessoa de Portugal com literacia financeira ALTA.
    - Conheces bem conceitos complexos de orçamento, poupança, crédito e planeamento financeiro pessoal, e compreendes as relações entre eles (ex: relação entre TAEG e TAN, impacto da inflação no valor real da poupança, diversificação e risco/retorno).
    - Tens memória e conhecimento sólido de valores concretos relevantes (ex: limites de dedução do IRS, percentagens, prazos legais).
    - Consegues identificar ambiguidades típicas de perguntas de literacia financeira, e raciocinas antes de responder.
    - Respondes com terminologia técnica correta e com confiança.
    - Escolhes sempre a opção mais correta, mesmo com ambiguidades.
    - Analisas a pergunta com cuidado e eliminas distratores implausíveis com base no teu conhecimento técnico.
    - Tens uma alta probabilidade de acertar, mas decides com base num raciocínio genuíno.
    """,
}

# shuffle options for simulated student
def shuffle_options(question):
    letters = ["A", "B", "C", "D"]
    texts = [question["options"][letter] for letter in letters]
    correct_text = question["options"][question["correct_answer"]]

    shuffled = texts
    random.shuffle(shuffled)
    new_correct_letter = letters[shuffled.index(correct_text)]

    fields = {
        "question": question["question"],
        "opt_a": shuffled[0],
        "opt_b": shuffled[1],
        "opt_c": shuffled[2],
        "opt_d": shuffled[3]
    }

    return fields, new_correct_letter


# simulate student answering question depending on their financial literacy level
def simulated_student(persona_system, question):
    # Send user request
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            persona_system
        ),
        (
            "user",
            """
                Pergunta: {question}
 
                Opções:
                A: {opt_a}
                B: {opt_b}
                C: {opt_c}
                D: {opt_d}
 
                Primeiro, retorna o texto exato da opção escolhida pela pessoa descrita acima.
                Depois, retorna a letra associada ao texto da opção escolhida.
                Considera todas as opções antes de responderes.
                Finalmente, em 1-2 frases, justifica a tua resposta tal como essa pessoa pensaria sobre
                esta pergunta (inclui se reconheces ou não os termos usados, e porquê).
            """
        )
    ])

    structured_model = model_persona.with_structured_output(StudentAnswer)

    chain = prompt | structured_model

    fields, new_correct_letter = shuffle_options(question)

    response = chain.invoke(fields)

    return response.chosen_option, new_correct_letter, response.reasoning

# estimate accuracy of answers per persona by running it more than once
def accuracy_persona(question, n_samples = 10):
    accuracy = {}
    reasonings = {}
    for persona, persona_system in PERSONAS.items():
        correct = 0
        reasoningList = []
        for i in range(0, n_samples):
            response, new_correct_letter, reasoning = simulated_student(persona_system, question)
            reasoningList.append(reasoning)
            if response[0] == new_correct_letter:
                correct += 1
        accuracy[persona] = correct/n_samples
        reasonings[persona] = reasoningList
    return accuracy, reasonings


DIFFICULTY_WEIGHTS = {"Baixo": 0.6, "Médio": 0.3, "Alto": 0.1}

# Turn the role-play accuracy results into a 1-3 difficulty score.
# 1 = easy (personas answer correctly often), 3 = hard (personas struggle).
def difficulty_from_accuracy(accuracy):
    ease = sum(DIFFICULTY_WEIGHTS[persona] * accuracy[persona] for persona in DIFFICULTY_WEIGHTS)
    score = 3 - ease * 2  # ease=1.0 (everyone gets it) -> 1 (easiest); ease=0.0 (no one gets it) -> 3 (hardest)
    return round(score, 2)

# Diversity Metrics: tag counter
def tag_diversity(quiz):
    return len({q["tag"] for q in quiz}) / len(quiz)

# Merge Ratings
def merge_ratings(question_evals, question_difficulty_scores, tag_diversity):
    def avg(criterion):
        return round(mean(qe[criterion]["score"] for qe in question_evals), 2)
 
    quiz_scores = {
        "topic_matching": avg("topic_matching"),
        "topic_grounding": avg("topic_grounding"),
        "answer_exclusivity": avg("answer_exclusivity"),
        "distractor_quality": avg("distractor_quality"),
        "hint_quality": avg("hint_quality"),
        "rationale_quality": avg("rationale_quality"),
        "tag_diversity": tag_diversity
    }
    all_scores = [
        quiz_scores["topic_matching"],
        quiz_scores["topic_grounding"],
        quiz_scores["answer_exclusivity"],
        quiz_scores["distractor_quality"],
        quiz_scores["hint_quality"],
        quiz_scores["rationale_quality"]
    ]
    quiz_scores["overall_score"] = round(mean(all_scores), 2)
    return quiz_scores    


# Pipeline

def read_quiz(q):
    return q['topics'], q['fin_lit_level'], q['source_content_by_tag'], q['questions']

 
def save_evaluation(quiz_obj, topics, level, evaluation):
    quiz_with_eval = dict(quiz_obj)
    quiz_with_eval["evaluation"] = evaluation.model_dump()
    output_file = "evaluated_quizzes/quiz_" + str(topics) + "_" + str(level) + ".json"
    output_file.replace(" ", "")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(quiz_with_eval, f, ensure_ascii=False, indent=2)

def pipeline_evaluation(quiz_obj):
    topics, level, source_content, quiz = read_quiz(quiz_obj)
    question_evals = []
    question_difficulty_scores = []
    question_results = []

    diversity = tag_diversity(quiz)

    for question in quiz:
        tag = question["tag"]
        source_content_tag = ""
        # check if there is source_content associated with the tag and merge this content to be used in evaluation
        content_quotes = source_content.get("geral", [])
        if not content_quotes:
            content_quotes = source_content.get(tag, [])
        if content_quotes:
            source_content_tag = "\n\n".join(
                s["content"] for s in content_quotes
            )

        # evaluate: tags/topic grounding, exclusivity, hint and rationale quality
        eval_q = evaluate_question_topic(topics, question, tag, source_content_tag)
        eval_q_dict = eval_q.model_dump()

        # evaluate: difficulty
        accuracy, reasonings = accuracy_persona(question)
        #difficulty = difficulty_from_accuracy(accuracy)

        question_evals.append(eval_q_dict)
        #question_difficulty_scores.append(difficulty)
        print(question["question"])
        print("\n")
        print(accuracy)

        question_results.append({
            "question": question["question"],
            "options": question["options"],
            "hint": question["hint"],
            "rationale": question["rationale"],
            "tag": question["tag"],
            "LLM_judge": eval_q_dict,
            "simulated_students": accuracy,
            "simulated_students_reasoning": reasonings,
        })

        print(eval_q_dict)

    quiz_results = merge_ratings(question_evals, question_difficulty_scores, diversity)

    return {
        "topics": topics,
        "fin_lit_level": level,
        "question_results": question_results,
        "quiz_results": quiz_results,
    }

    #save_evaluation(quiz_obj, topics, level, evaluation)

def load_quizzes_from_folder(folder):
    quizzes = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            quizzes.extend(obj)
        else:
            quizzes.append(obj)
    return quizzes

def save_evaluation(result):
    output_file = "evaluated_quizzes/quiz_" + str(result["topics"]) + "_" + str(result["fin_lit_level"]) + ".json"
    output_file.replace(" ", "")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def main():
    quizzes = load_quizzes_from_folder("generated_quizzes")
    for q in quizzes:
        result = pipeline_evaluation(q)
        save_evaluation(result)
        print("Evaluation saved!\n")


if __name__ == "__main__":
    main()

