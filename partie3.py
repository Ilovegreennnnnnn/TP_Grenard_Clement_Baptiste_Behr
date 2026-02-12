import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client, Evaluation
from partie2 import plan_weekly_menu 

load_dotenv()

groq_client = Groq()
langfuse = get_client()

# 3.1 - CRÉATION DU DATASET

def create_chefbot_dataset():
    dataset_name = "chefbot-menu-eval-baptiste-clement"
    
    try:
        langfuse.get_dataset(dataset_name)
        print(f"Le dataset '{dataset_name}' existe déjà.")
    except:
        dataset = langfuse.create_dataset(
            name=dataset_name,
            description="Évaluation des menus hebdomadaires de ChefBot",
            metadata={"version": "1.0"}
        )

        test_cases = [
            {
                "input": {"constraints": "Repas pour diabétique"},
                "expected": {"must_avoid": ["sucre", "pâtes blanches", "sirop"], "must_include": ["légumes verts", "fibres"]}
            },
            {
                "input": {"constraints": "Régime sans gluten, budget serré"},
                "expected": {"must_avoid": ["blé", "farine de blé", "pâtes"], "must_include": ["riz", "légumineuses"]}
            },
            {
                "input": {"constraints": "Sportif, haute teneur en protéines, pas de produits laitiers"},
                "expected": {"must_avoid": ["lait", "fromage", "beurre"], "must_include": ["poulet", "oeufs", "lentilles"]}
            },
            {
                "input": {"constraints": "Étudiant, cuisine avec seulement un micro-ondes, 20€/semaine"},
                "expected": {"must_avoid": ["four", "plaques de cuisson"], "must_include": ["conserves", "pommes de terre"]}
            },
            {
                "input": {"constraints": "Menu traditionnel français pour 6 convives, saison hiver"},
                "expected": {"must_avoid": ["tomates", "fraises"], "must_include": ["vin rouge", "viande en sauce", "racines"]}
            }
        ]

        for case in test_cases:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=case["input"],
                expected_output=case["expected"]
            )
        print(f"✓ Dataset '{dataset_name}' créé avec {len(test_cases)} cas.")

# 3.2 - ÉVALUATEUR PROGRAMMATIQUE

def rule_evaluator(output: str, expected: dict) -> dict:
    """Vérification stricte des mots-clés (Rule-based)."""
    scores = {}
    output_lower = output.lower()

    avoid_items = expected.get("must_avoid", [])
    found_forbidden = [item for item in avoid_items if item.lower() in output_lower]
    scores["safety_score"] = 1.0 if not found_forbidden else 0.0

    include_items = expected.get("must_include", [])
    if not include_items:
        scores["inclusion_score"] = 1.0
    else:
        found_required = [item for item in include_items if item.lower() in output_lower]
        scores["inclusion_score"] = len(found_required) / len(include_items)

    return scores

# 3.3 - LLM JUGE

JUDGE_PROMPT = """Tu es un critique gastronomique expert. Évalue le menu suivant sur une échelle de 0.0 à 1.0.

Critères :
1. pertinence : Le menu respecte-t-il strictement les contraintes de l'utilisateur ?
2. creativite : Les plats sont-ils variés et originaux ?
3. praticite : Les recettes sont-elles réalisables par un amateur ?

Réponds UNIQUEMENT en JSON :
{
    "pertinence": 0.0,
    "creativite": 0.0,
    "praticite": 0.0,
    "explanation": "justification courte"
}"""

@observe(name="llm-judge", as_type="generation")
def llm_judge(question: str, output: str, expected: dict) -> dict:
    """Utilisation d'un LLM pour évaluer la qualité sémantique."""
    user_content = f"Contraintes: {question}\n\nMenu généré: {output}"
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# 3.4 - LANCEMENT DE L'EXPÉRIENCE

def run_chefbot_experiment():
    dataset = langfuse.get_dataset("chefbot-menu-eval-baptiste-clement")

    def task(item):
        constraints = item.input["constraints"]
        result = plan_weekly_menu(constraints)
        return result.get("menu", "")

    def evaluator(**kwargs):
        output = kwargs.get("output")
        expected = kwargs.get("expected_output")
        input_data = kwargs.get("input")

        prog_scores = rule_evaluator(output, expected)
        
        llm_scores = llm_judge(input_data["constraints"], output, expected)

        return [
            Evaluation(name="safety_rule", value=prog_scores["safety_score"]),
            Evaluation(name="inclusion_rule", value=prog_scores["inclusion_score"]),
            Evaluation(name="llm_pertinence", value=llm_scores["pertinence"], comment=llm_scores["explanation"]),
            Evaluation(name="llm_creativite", value=llm_scores["creativite"]),
            Evaluation(name="llm_praticite", value=llm_scores["praticite"]),
        ]

    langfuse.run_experiment(
        name=f"chefbot-eval-{datetime.now().strftime('%H%M%S')}",
        data=dataset.items,
        task=task,
        evaluators=[evaluator],
        metadata={
            "model": "llama-3.1-8b-instant",
            "partie": "3"
        }
    )

if __name__ == "__main__":
    create_chefbot_dataset()
    run_chefbot_experiment()
    langfuse.flush()