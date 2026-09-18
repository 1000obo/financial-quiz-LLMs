import os
import json
import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain.agents.structured_output import ToolStrategy
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initializing the model
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key = api_key
)

model_generation = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    api_key = api_key
)

TOPIC_TAGS = {
    "Mindset, Comportamento e Hábitos": [
        "Mentalidade financeira",
        "Crenças financeiras",
        "Valores pessoais",
        "Hábitos financeiros",
        "Comportamento financeiro",
        "Identidade financeira",
        "Autocontrolo e disciplina",
        "Propósito financeiro",
    ],
    "Orçamento e Gestão Corrente": [
        "Orçamento pessoal",
        "Tipos de despesa",
        "Controlo financeiro",
        "Métodos de orçamentação",
        "Eficiência orçamental",
        "Registo e categorização de despesas",
        "Objetivos financeiros",
    ],
    "Compras e Poupança": [
        "Consumo consciente",
        "Compras impulsivas",
        "Planeamento de compras",
        "Poupança automática",
        "Fundo de emergência",
        "Regra 50/30/20",
        "Micro-poupanças",
        "Literacia financeira familiar",
    ],
    "Crédito e Gestão de Dívida": [
        "Crédito responsável",
        "Endividamento",
        "Taxas de juro",
        "Gestão da dívida",
        "BNPL (buy now pay later)",
        "TAEG e MTIC",
        "Taxa de esforço",
        "Renegociação e consolidação",
    ],
    "Investimento (básico) e Produtos com Garantia de Capital": [
        "Conceitos de investimento",
        "Inflação",
        "Juro composto",
        "Produtos com garantia",
        "Rentabilidade e risco",
        "Liquidez",
        "Poupança e investimento",
        "Fiscalidade dos investimentos",
        "Cálculo de rendimentos",
    ],
    "Investimento (avançado) e Produtos sem Garantia de Capital": [
        "Mercados financeiros",
        "Ações e obrigações",
        "Fundos, ETFs e REITs",
        "Diversificação e risco",
        "Forex e metais preciosos",
        "Criptomoedas",
        "Investimento ESG",
        "Estratégias FIRE",
    ],
    "Planeamento de Médio-Longo Prazo e Carteiras de Investimento": [
        "Planeamento financeiro",
        "Definição de objetivos",
        "Perfil de investidor",
        "Diversificação de carteiras",
        "Alocação de ativos",
        "Reajustamento de carteiras",
        "Horizonte temporal",
        "Eficiência fiscal",
    ],
    "Fiscalidade e Otimização Financeira": [
        "Impostos pessoais",
        "IRS e deduções",
        "IMI, IMT e IUC",
        "IVA",
        "Otimização fiscal",
        "Faturas e e-fatura",
        "Benefícios fiscais",
        "PPR e fiscalidade",
    ],
}

 
def _normalize(s: str) -> str:
    return s.strip().lower()
 
 
def get_tags_topics(topics):
    # Collects tags for the given list of topics
    normalized_map = {_normalize(k): v for k, v in TOPIC_TAGS.items()}
    tags = []
    for topic in topics:
        topic_tags = normalized_map.get(_normalize(topic), [])
        for tag in topic_tags:
            if tag not in tags:
                tags.append(tag)
    return tags


# Data Structures
class QuizOptions(BaseModel):
    A: str
    B: str
    C: str
    D: str

class QuizQuestion(BaseModel):
    question: str
    options: QuizOptions
    correct_answer: str
    hint: str
    rationale: str
    tag: str

class Quiz(BaseModel):
    questions: List[QuizQuestion]
    topics: List[str]
    level: str
    articles: List[str]

# Question/response pair (stem + correct answer only, no distractors yet)
class StemKey(BaseModel):
    tag: str
    question: str
    correct_answer: str
    hint: str
    rationale: str

class StemKeyBatch(BaseModel):
    items: List[StemKey]

# Distractors (wrong options) generated for a stem, keyed by its position in the batch
class Distractors(BaseModel):
    index: int  # position in the StemKeyBatch this belongs to
    B: str
    C: str
    D: str

class DistractorBatch(BaseModel):
    items: List[Distractors]


# Code
def prepare_data():
    # Load Files
    docs = load_files("converted_txt")
    # Split the text into chunks
    chunks = split_text(docs)
    # Embeddings and Indexing
    vectorstore = embeddings_indexing(chunks)
    return vectorstore

def load_files(folder):
    documents = []
    for file in Path("./" + folder).glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={"filename": str(file)}
            )
        )
    return documents

def split_text(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

def embeddings_indexing(chunks):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    # We take the text chunks created and the initalized embedding model, and build a vector database.
    # Embedding: The chunks are sent to OpenAI, which returns a long vector of numbers (a numerical representation) for each chunk.
    # Indexing: Chroma takes those numerical representations and maps them into a library so the system can find related topics given their mathematical similarity.
    # Chroma is an open-source vector database designed specifically for AI applications to store, index, and query embeddings (numerical representations of data).
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )
    return vectorstore

# Obtain relevant knowledge from articles/newsletters given topic and tags (semantic RAG over .txt files)
def search_knowledge(topics, tags, vectorstore):
    results_by_tag = {}
    if tags:
        for tag in tags:
            seen = set()
            docs = vectorstore.similarity_search(f"Informação sobre {tag}", k=4)
            unique_docs = []
            for d in docs:
                key = (d.metadata.get("filename"), d.page_content[:200])
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(d)
            results_by_tag[tag] = unique_docs
    else:
        results_by_tag["geral"] = vectorstore.similarity_search(
            "Informação sobre " + ", ".join(topics), k=10
        )
    return results_by_tag


DIFFICULTY_SUMMARY = {
    "Baixo": (
        "Foca-te em ideias práticas e consequências do dia-a-dia, não em mecanismos técnicos. "
        "Preserva factos e números tal como estão, mas não os transformes em fórmulas ou rácios."
    ),
    "Médio": (
        "Inclui números, percentagens e relações simples de causa-efeito sempre que existam na fonte, "
        "para permitir perguntas de cálculo simples ou de aplicação de um conceito. "
        "Explica juros simples, prestação mensal e taxa de esforço com detalhe suficiente para serem aplicados."
    ),
    "Alto": (
        "Preserva detalhes técnicos, números, condições, prazos, exceções ('desde que', 'exceto quando') "
        "e relações causais completas. Explica como os conceitos se relacionam entre si "
        "(ex.: inflação vs. valor real da poupança; diversificação vs. risco/retorno), "
        "não apenas cada um isoladamente. Garante dados suficientes para cálculos com várias etapas."
    ),
}

# Summarize relevant knowledge (appropriate to the topic and difficulty) to send to the quiz generator
def summarize_relevant_knowledge(search, topics, tags, level):
    
    # prepare the relevant knowledge
    context = "\n\n".join(
        s.page_content for s in search
    )

    SYSTEM_PROMPT = """És um especialista em criar resumos em Português Europeu (PT-PT) para a geração de questionários de literacia financeira e educação. 
    Analisa a documentação fornecida que foi extraída de acordo com uma lista de tópicos e sub-tópicos (tags) e cria um resumo claro 
    que possa ser usado num gerador de questionários. Os questionários têm como base os tópicos, as tags e o nível financeiro do utilizador. 
    
    Segue as seguintes regras:

    GERAL:
    - Os resumos devem estar em Português Europeu (PT-PT). Não uses Português do Brasil.

    CONTEÚDO E ORGANIZAÇÃO:
    - Usa apenas a informação fornecida. Não inventes informação para além da que foi fornecida.
    - Preserva números, condições, causas e fontes citadas (ex: "segundo esta entidade") — não os substituas por generalizações vagas.
    - Identifica definições, relações, comparações, valores, e exemplos importantes.
    - Inclui o "porquê"/"como", não só o "o quê"; não reduzas cada tag a uma frase-conclusão.
    - Dá prioridade a conceitos que possam ser transformados em perguntas para questionários.

    TAGS:
    - Organiza o resumo por tag sempre que possível.
    - Se o conteúdo de uma tag não for diretamente relevante ao tema dessa tag, escreve exatamente: "Sem informação suficiente nas fontes fornecidas."

    DIFICULDADE: 
    - Adapta a profundidade e o grau de detalhe técnico do resumo ao nível de dificuldade indicado, para que o resumo forneça material suficiente e adequado à construção de perguntas desse nível. """

    # Send user request
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            SYSTEM_PROMPT
        ),
        (
            "user",
            """
                Resume o seguinte conteúdo em Português Europeu (PT-PT) para ser usado num gerador de questionários:

                Tópicos:
                {topics}

                Tags:
                {tags}

                Nível de literacia financeira:
                {level}

                Critério de dificuldade a aplicar ao resumo:
                {difficulty}

                Conteúdo:
                {context}
            """
        )
    ])

    chain = prompt | model

    response = chain.invoke({
        "topics": topics,
        "tags": tags,
        "level": level,
        "difficulty": DIFFICULTY_SUMMARY[level],
        "context": context
    })

    return response.content


DIFFICULTY = {
    "Baixo": (
        "A pergunta deve ser respondível com senso comum e experiência do dia-a-dia, "
        "NUNCA com conhecimento financeiro, mesmo básico. Exclui mecânica de crédito, "
        "juros ou cálculos. NÃO uses jargão financeiro, siglas, ou o nome técnico da tag."
        "Testa reconhecimento de uma ideia prática."
    ),
    "Médio": (
        "A pergunta deve exigir aplicar um conceito financeiro básico (orçamento, poupança, "
        "crédito simples) a uma situação concreta — "
        "nunca apenas repetir uma definição. Cálculos simples (1-2 operações) são bem-vindos "
        "quando a fonte fornece os dados."
    ),
    "Alto": (
        "A pergunta TEM de exigir relacionar pelo menos dois conceitos entre si (ex.: risco vs. "
        "liquidez, TAEG vs. TAN, inflação vs. valor real da poupança) ou aplicar um critério "
        "técnico/condição a uma situação concreta. Não deve ser respondível apenas pelo título do "
        "tópico nem por senso comum — exige ter compreendido o detalhe da fonte. Usa cálculos "
        "multi-etapa ou comparação de cenários sempre que a fonte o permita."
    ),
}

# Generate questions given summary and difficulty level
def generate_questions(summary, topics, tags, level, articles):
    
    SYSTEM_PROMPT = """És um especialista em criar perguntas de literacia financeira e educação em Português Europeu (PT-PT).
    A tua ÚNICA tarefa é gerar o par (pergunta, resposta correta) para um questionário com 10 (dez) perguntas de escolha múltipla
    — NÃO geres opções erradas, isso é feito noutro passo.
    Tens acesso a conteúdo sumarizado de artigos que respeita os tópicos, as tags e nível financeiro do utilizador. 
    
    Segue as seguintes regras:

    GERAL: 
    - Gera EXATAMENTE 10 (dez) perguntas.
    - Os questionários devem estar em Português Europeu (PT-PT). Não uses Português do Brasil.

    CONTEÚDO:
    - Usa exclusivamente o conteúdo fornecido; nunca completes com conhecimento geral.
    - Os questionários devem refletir com precisão as informações de referência, e a resposta correta deve ser bem fundamentada de acordo com essa informação.

    TAGS: 
    - Se uma tag disser "Sem informação suficiente nas fontes fornecidas" ou não tiver base para uma pergunta fundamentada, salta-a e gera antes outra pergunta (repetindo tags válidas se preciso).
    - Ignorar por completo tags sem cobertura no resumo.
    - Cada pergunta deve ter associada exatamente uma tag, escolhida obrigatoriamente da lista de tags (não inventes tags novas).
    - Distribui as perguntas pelas tags de forma equilibrada, cobrindo o maior número possível de tags distintas antes de repetir alguma — mas a fundamentação na fonte tem sempre prioridade sobre a cobertura de tags.
    
    FORMATO DAS QUESTÕES:
    - Varia o formato e a abertura das perguntas ao longo do questionário. No máximo 2-3 perguntas podem começar pela mesma fórmula (ex.: "Qual é...").

    FORMATO DA RESPOSTA CORRETA:
    - 'correct_answer' deve ser uma opção de resposta curta e autossuficiente (a frase que apareceria ao lado de "A)" numa escolha múltipla) — NÃO uma explicação. Alvo: 6-16 palavras. Toda a justificação/detalhe extra vai para 'rationale', não para 'correct_answer'.
    - 'hint': aponta o conceito, categoria ou tipo de raciocínio/cálculo a usar, sem revelar a resposta (ex. o que acontece ao dinheiro que não gastas por impulso ao longo de um ano?).
    - 'rationale': em 1-2 frases, explica porque a resposta está certa com o conteúdo da fonte (não é necessário citar) — não te limites a reformular a resposta correta por outras palavras.
    - Estes campos devem estar em Português Europeu (PT-PT). Não uses Português do Brasil.    
    
    DIFICULDADE:
    - Respeita rigorosamente o critério de dificuldade fornecido para construir a pergunta (aplica-se apenas à pergunta, não às opções, que são geradas noutro passo).
    - Usa frases para completar, cenários aplicados e perguntas de cálculo de acordo com o nível de dificuldade definido.
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
                Gera os pares (pergunta, resposta correta) em Português Europeu (PT-PT) para um questionário com 10 (dez) perguntas de acordo com a informação fornecida:
                
                Tópicos:
                {topics}

                Tags:
                {tags}

                Nível de literacia financeira:
                {level}

                Critério de dificuldade a aplicar à construção das perguntas:
                {difficulty}

                Artigos:
                {articles}

                Conteúdo:
                {summary}
            """
        )
    ])

    structured_model = model_generation.with_structured_output(StemKeyBatch)

    chain = prompt | structured_model

    response = chain.invoke({
        "topics": topics,
        "tags": tags,
        "level": level,
        "difficulty": DIFFICULTY[level],
        "articles": articles,
        "summary": summary
    })

    return response.items


'''DISTRACTOR_THEORY = """Regras de Distractor Theory (aplica-se a TODOS os níveis):
- Cada distrator deve ser mutuamente exclusivo dos outros (sem sobreposição de significado).
- Cada distrator deve ser gramaticalmente consistente com a pergunta.
- As 3 opções erradas devem ter o MESMO grau de detalhe, extensão e certeza da resposta correta — nunca mais vagas, curtas ou hesitantes só por serem erradas.
- Evita termos absolutos (\"sempre\", \"nunca\", \"todos\", \"nenhum\") nas opções — são facilmente descartados por eliminação.
- Evita termos vagos ou ambíguos.
- Nenhum distrator pode ser uma combinação de outras opções de resposta.
- Gera as 3 opções como um conjunto (pensa nelas em simultâneo), não uma a uma isoladamente.
"""'''

DISTRACTOR_THEORY = """Regras de Distractor Theory (aplica-se a TODOS os níveis):
1. Cada distrator deve ser mutuamente exclusivo dos outros (sem sobreposição de significado).
2. Cada distrator deve ser gramaticalmente consistente com a pergunta e resposta correta
3. As 3 opções erradas devem ter o MESMO grau de detalhe, extensão e certeza da resposta certa — nunca mais vagas, curtas, ou hesitantes só por serem erradas.
4. As 3 opções erradas devem ter a MESMA estrutura gramatical que a resposta certa — nunca se devem destacar por serem diferentes gramaticalmente ou em termos de pontuação.
5. Evita termos absolutos ("sempre", "nunca", "todos", "nenhum", "apenas") nas opções — são facilmente descartados por eliminação.
6. Evita termos vagos ou ambíguos.
7. Nenhum distrator pode ser uma combinação de outras opções de resposta.
8. Gera as 3 opções como um conjunto (pensa nelas em simultâneo), não uma a uma isoladamente.
9. REGRA DE NÃO-AMBIGUIDADE (aplica-se mesmo respeitando as regras acima): cada distrator tem de ser objetivamente incorreto face à fonte fornecida. Um distrator convincente é desejável; um distrator defensável como resposta alternativa correta não é.
"""

DISTRACTOR_CONSTRUCTION_BY_LEVEL = {
    "Baixo": (
        "Os distratores devem ser plausíveis para uma pessoa comum mas rejeitáveis "
        "por bom senso — pelo menos um deve refletir um erro prático comum do dia-a-dia. "
        "Não uses distratores absurdos ou fisicamente impossíveis."
    ),
    "Médio": (
        "Os distratores devem estar diretamente relacionados com o conceito perguntado. "
        "Pelo menos um deve representar um erro comum de raciocínio financeiro "
        "(ex.: confundir poupança com dinheiro disponível para gastar), não uma opção irrelevante."
    ),
    "Alto": (
        "TODOS os distratores devem falar diretamente do mesmo assunto/tag da pergunta — nenhum "
        "pode ser descartado por ser visivelmente irrelevante ao tema. Constrói-os com estes métodos, "
        "sempre que a fonte o permita:\n"
        "1) TROCA DE DETALHE — pega num facto real da fonte e altera um número, prazo ou condição.\n"
        "2) CONFUSÃO ENTRE CONCEITOS PRÓXIMOS — aplica um princípio verdadeiro ao conceito errado "
        "(ex.: lógica da liquidez numa pergunta sobre rentabilidade).\n"
        "3) CAUSA/EFEITO INVERTIDOS OU INCOMPLETOS — apresenta a relação certa invertida, exagerada "
        "ou só parcialmente certa.\n"
        "Usa APENAS UM destes métodos por distrator — não combines dois ou mais métodos na mesma "
        "opção. Combinar métodos tende a produzir distratores tão próximos da resposta correta que "
        "deixam de ser objetivamente incorretos, o que viola a regra de não-ambiguidade."
    ),
}

# Distractor module: given the (pergunta, resposta correta) pairs from generate_stems,
# generates exactly 3 distratores (opções erradas B, C, D) for each one.
def generate_distractors(stems, topics, tags, level, summary):

    SYSTEM_PROMPT = """És um especialista em criar distratores (opções erradas) para perguntas de escolha múltipla de literacia financeira, em Português Europeu (PT-PT). Não uses Português do Brasil.

    Vais receber uma lista de pares (pergunta, resposta correta) já fundamentados numa fonte, o context de referência e o nível de literacia do utilizador.
    Para CADA item, gera exatamente 3 distratores (B, C, D) plausíveis, mantendo o 'index' do item original para os poderes associar de volta.

    {distractor_theory}

    Critério de construção específico para este nível de dificuldade:
    {construction}
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
                Gera os distratores (opções B, C, D) em Português Europeu (PT-PT) para os seguintes pares pergunta/resposta:

                Nível de literacia financeira:
                {level}

                Pares pergunta/resposta (index, tag, pergunta, resposta correta):
                {items}

                Contexto de referência (para garantir que os distratores permanecem plausíveis e ligados ao tema):
                {summary}
            """
        )
    ])

    items_text = "\n\n".join(
        f"[{i}] Tag: {s.tag}\nPergunta: {s.question}\nResposta correta: {s.correct_answer}"
        for i, s in enumerate(stems)
    )

    structured_model = model_generation.with_structured_output(DistractorBatch)

    chain = prompt | structured_model

    response = chain.invoke({
        "distractor_theory": DISTRACTOR_THEORY,
        "construction": DISTRACTOR_CONSTRUCTION_BY_LEVEL[level],
        "level": level,
        "items": items_text,
        "summary": summary
    })

    return {d.index: d for d in response.items}

# Combine the question/response pairs with their distractors into full QuizQuestion objects.
# Drops any stem that didn't get matching distractors back, rather than shipping a broken question.
def _assemble_quiz_questions(stems, distractors):
    questions = []
    for i, s in enumerate(stems):
        d = distractors.get(i)
        if d is None:
            continue
        questions.append(QuizQuestion(
            question=s.question,
            options=QuizOptions(A=s.correct_answer, B=d.B, C=d.C, D=d.D),
            correct_answer="A",
            hint=s.hint,
            rationale=s.rationale,
            tag=s.tag
        ))
    return questions

# Orchestrates the two modules above: question/response pairs first, then distractors,
# then assembles everything into a Quiz.
def generate_quiz(summary, topics, tags, level, articles):
    stems = generate_questions(summary, topics, tags, level, articles)
    distractors = generate_distractors(stems, topics, tags, level, summary)
    questions = _assemble_quiz_questions(stems, distractors)
    return Quiz(questions=questions, topics=topics, level=level, articles=articles)

def pipeline_quiz(topics, level, vectorstore):
    tags = get_tags_topics(topics)
    content_by_tag = search_knowledge(topics, tags, vectorstore)
    all_docs = [d for docs in content_by_tag.values() for d in docs]
    articles = list(set(d.metadata["filename"].replace("converted_txt/", "") for d in all_docs))
    summary = summarize_relevant_knowledge(all_docs, topics, tags, level)
    quiz = generate_quiz(summary, topics, tags, level, articles)

    return quiz, content_by_tag, summary

def save_quiz(quiz, content_by_tag, summary):
    all_results = []
    output_file = "generated_quizzes/quiz_" + str(quiz.topics) + "_" + str(quiz.level) + ".json"
    output_file.replace(" ", "")
    if quiz:
        all_results.append({
            "articles": quiz.articles,
            "topics": quiz.topics,
            "fin_lit_level": quiz.level,
            "source_content_by_tag": {
                tag: [{"filename": d.metadata.get("filename"), "content": d.page_content} for d in docs]
                for tag, docs in content_by_tag.items()
            },
            "summary_content": summary,
            "questions": [q.model_dump() for q in quiz.questions]
        })

        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

def run_quiz(quiz):
    score = 0

    print("\n=== Início do Quiz ===\n")

    for i, q in enumerate(quiz.questions, start=1):
        print(f"\nPergunta {i}/{len(quiz.questions)}  [Tag: {q.tag}]")
        print("-" * 50)
        print(q.question)

        # save options in list
        options = [
            ("A", q.options.A),
            ("B", q.options.B),
            ("C", q.options.C),
            ("D", q.options.D)
        ]

        # save correct answer before shuffling - should be A
        correct_text = dict(options)[q.correct_answer]

        # shuffle
        random.shuffle(options)

        # criar novas letras A-D
        shuffled = {}
        letters = ["A", "B", "C", "D"]

        for letter, (_, text) in zip(letters, options):
            shuffled[letter] = text

        # find new correct position
        new_correct_answer = [
            letter for letter, text in shuffled.items()
            if text == correct_text
        ][0]

        # print options
        for letter, text in shuffled.items():
            print(f"{letter}) {text}")

        # pedir resposta ou pista
        while True:
            answer = input("\nEscolhe uma opção (A/B/C/D) ou escreve H para pista: ").strip().upper()

            if answer == "H":
                print(f"\nPista: {q.hint}")
                continue

            if answer in ["A", "B", "C", "D"]:
                break

            print("Opção inválida. Tenta novamente.")

        # verificar resposta
        if answer == new_correct_answer:
            print("\nCorreto!")
            score += 1
        else:
            print(f"\nErrado! A resposta correta era: {new_correct_answer}")

        # mostrar explicação
        print(f"\nExplicação: {q.rationale}")

        input("\nCarrega ENTER para continuar...")

    print("\n=== Resultado ===")
    print(f"Pontuação: {score}/{len(quiz.questions)}")

    percentage = score / len(quiz.questions) * 100
    print(f"Percentagem: {percentage:.1f}%")


def main():
    vectorstore = prepare_data()
    topics = TOPIC_TAGS.items()
    levels = ["Baixo", "Médio", "Alto"]

    for t in topics:
        for l in levels:
            # Generate quiz
            quiz, content_by_tag, summary = pipeline_quiz([t[0]], l, vectorstore)
            # Save quiz
            save_quiz(quiz, content_by_tag, summary)
            print ("Quiz Saved!")
    
    '''while True:
        print("\n=== Gerador de Quizzes Financeiros ===")

        # Ask topics
        topics_input = input(
            "Coloca os tópicos separados por ponto e vírgula (ex., Compras e Poupança; Crédito e Gestão de Dívida): "
        )
        topics = [topic.strip() for topic in topics_input.split(";")]

        # Ask level
        while True:
            level = input("Coloca o teu nível de literacia financeira (Baixo/Médio/Alto): ").strip()

            if level in ["Baixo", "Médio", "Alto"]:
                break

            print("Nível inválido. Por favor, escolhe 'Baixo', 'Médio' ou 'Alto'.")

        # Generate quiz
        quiz, content_by_tag, summary = pipeline_quiz(topics, level, vectorstore)

        # Save quiz
        save_quiz(quiz, content_by_tag, summary)

        # Run quiz
        run_quiz(quiz)

        # Repeat?
        repeat = input("\nQueres gerar outro quiz? (s/n): ").strip().lower()

        if repeat not in ["y", "yes", "s", "sim"]:
            print("Adeus!")
            break
    '''
    

if __name__ == "__main__":
    main()