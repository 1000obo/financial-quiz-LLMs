import os
import json
import pandas as pd
import random

def load_quizzes(folder):
    results = []
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results

def build_question_rows(results, selected_num = 5):
    """One row per question: for finer-grained expert comparison against the LLM judge."""
    rows = []
    question_id = 0
    for r in results:
        #r = r[0]
        idx = 1
        topics = ", ".join(r["topics"]) if isinstance(r["topics"], list) else r["topics"]
        selected = random.sample(range(1, 11), selected_num)
        for pq in r["question_results"]:
            if idx in selected:
                row = {
                    "ID": question_id,
                    "Tópico": topics,
                    "Tag": pq["tag"],
                    "Pergunta": pq["question"],
                    "A (certa)": pq["options"]["A"],
                    "B": pq["options"]["B"],
                    "C": pq["options"]["C"],
                    "D": pq["options"]["D"],
                    "Dica": pq["hint"],
                    "Racional": pq["rationale"],
                    "Adequação ao Tópico e Tags": "",
                    "Fundamentação e Correção": "",
                    "Exclusividade das Respostas": "",
                    "Qualidade dos Distratores": "",
                    "Qualidade da Dica": "",
                    "Qualidade do Racional": "",
                    "Comentários": "",
                }
                rows.append(row)
            idx += 1
            question_id += 1
    return pd.DataFrame(rows)

def build_question_rows_difficulty(results, selected_num = 5):
    """One row per question: for finer-grained expert comparison against the LLM judge."""
    rows = []
    question_id = 0
    for r in results:
        #r = r[0]
        idx = 1
        topics = ", ".join(r["topics"]) if isinstance(r["topics"], list) else r["topics"]
        selected = random.sample(range(1, 11), selected_num)
        for pq in r["question_results"]:
            if idx in selected:
                row = {
                    "ID": question_id,
                    "Tópico": topics,
                    "Tag": pq["tag"],
                    "Pergunta": pq["question"],
                    "A (certa)": pq["options"]["A"],
                    "B": pq["options"]["B"],
                    "C": pq["options"]["C"],
                    "D": pq["options"]["D"],
                    "Dificuldade da Pergunta": "",
                    "Comentários": "",
                }
                rows.append(row)
            idx += 1
            question_id += 1
    return pd.DataFrame(rows)

def export_workbook(results, output_path = "quizzes_for_experts_clean.xlsx"):
    question_df = build_question_rows(results)
    question_df_difficulty = build_question_rows_difficulty(results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        question_df.to_excel(writer, sheet_name="Por Topico", index=False)
        question_df_difficulty.to_excel(writer, sheet_name="Por Dificuldade", index=False)

    return output_path

def build_quiz_summary_rows_AI_evaluated(results):
    """One row per quiz: automated scores."""
    rows = []
    for r in results:
        topics = ", ".join(r["topics"]) if isinstance(r["topics"], list) else r["topics"]
        row = {
            "topics": topics,
            "level": r["fin_lit_level"],
            **r["quiz_results"]
        }
        rows.append(row)
    return pd.DataFrame(rows)

def build_question_rows_AI_evaluated(results):
    """One row per question: for finer-grained expert comparison against the expert judge."""
    rows = []
    question_id = 0
    for r in results:
        topics = ", ".join(r["topics"]) if isinstance(r["topics"], list) else r["topics"]
        for pq in r["question_results"]:
            row = {
                "ID": question_id,
                "topics": topics,
                "tag": pq["tag"],
                "level": r["fin_lit_level"],
                "question": pq["question"],
                "option_A": pq["options"]["A"],
                "option_B": pq["options"]["B"],
                "option_C": pq["options"]["C"],
                "option_D": pq["options"]["D"],
                "hint": pq["hint"],
                "rationale": pq["rationale"],
                "topic_matching": pq["LLM_judge"]["topic_matching"]["score"],
                "topic_matching_comment": pq["LLM_judge"]["topic_matching"]["justification"],
                "topic_grounding": pq["LLM_judge"]["topic_grounding"]["score"],
                "topic_grounding_comment": pq["LLM_judge"]["topic_grounding"]["justification"],
                "answer_exclusivity": pq["LLM_judge"]["answer_exclusivity"]["score"],
                "answer_exclusivity_comment": pq["LLM_judge"]["answer_exclusivity"]["justification"],
                "distractor_quality": pq["LLM_judge"]["distractor_quality"]["score"],
                "distractor_quality_comment": pq["LLM_judge"]["distractor_quality"]["justification"],
                "hint_quality": pq["LLM_judge"]["hint_quality"]["score"],
                "hint_quality_comment": pq["LLM_judge"]["hint_quality"]["justification"],
                "rationale_quality": pq["LLM_judge"]["rationale_quality"]["score"],
                "rationale_quality_comment": pq["LLM_judge"]["rationale_quality"]["justification"],
                "simulated_low": pq["simulated_students"]["Baixo"],
                "simulated_medium": pq["simulated_students"]["Médio"],
                "simulated_high": pq["simulated_students"]["Alto"],
                "simulated_low_reasoning": pq["simulated_students_reasoning"]["Baixo"],
                "simulated_medium_reasoning": pq["simulated_students_reasoning"]["Médio"],
                "simulated_high_reasoning": pq["simulated_students_reasoning"]["Alto"],
            }
            rows.append(row)
            question_id += 1
    return pd.DataFrame(rows)

def export_evaluated_workbook(results, output_path = "AI_evaluated_quizzes_clean.xlsx" ):
    question_df = build_question_rows_AI_evaluated(results)
    quiz_df = build_quiz_summary_rows_AI_evaluated(results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        quiz_df.to_excel(writer, sheet_name="Por Quiz", index=False)
        question_df.to_excel(writer, sheet_name="Por Pergunta", index=False)
    
    return output_path


def main():
    results = load_quizzes("evaluated_quizzes")
    path = export_workbook(results)
    print(f"Workbook written to {path}")
    results = load_quizzes("evaluated_quizzes")
    path = export_evaluated_workbook(results)
    print(f"AI-evaluated workbook written to {path}")


if __name__ == "__main__":
    main()