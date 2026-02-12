import os
import json
from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client

load_dotenv()

groq_client = Groq()
langfuse = get_client()

# FONCTIONS DE SOUS-ÉTAPES

@observe(name="planning", as_type="generation")
def _plan_steps(constraints: str, retry: bool = True) -> dict:
    """Étape 1 : Planification"""
    
    prompt = f"""Tu es un planificateur culinaire expert. 
    Décompose la création d'un menu hebdomadaire selon ces contraintes : {constraints}.
    Génère 3 étapes concrètes (ex: 'identifier les protéines', 'choisir les légumes', 'équilibrer les repas').
    
    RETOURNE UNIQUEMENT DU JSON au format suivant :
    {{
        "steps": ["étape 1", "étape 2", "étape 3"],
        "reasoning": "explication courte"
    }}"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        if retry:
            get_client().update_current_span(
                level="ERROR",
                status_message=f"JSON invalide, retry en cours... {str(e)}"
            )
            return _plan_steps(constraints, retry=False)
        else:
            get_client().update_current_span(
                level="ERROR",
                status_message="Échec définitif du parsing JSON."
            )
            raise e

@observe(name="execute-step", as_type="generation")
def _execute_step(step_description: str, step_index: int, previous_results: list) -> dict:
    """Étape 2 : Exécution d'une étape spécifique"""
    
    context_str = "\n".join([f"- {r['output']}" for r in previous_results]) if previous_results else "Aucun"
    
    prompt = f"""Exécute cette étape : {step_description}
    Contexte précédent : {context_str}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return {
        "step": step_description,
        "output": response.choices[0].message.content
    }

@observe(name="synthesis", as_type="generation")
def _synthesize_menu(constraints: str, results: list) -> str:
    """Étape 3 : Synthèse finale"""
    
    all_work = "\n\n".join([f"RÉSULTAT {r['step']} :\n{r['output']}" for r in results])
    
    prompt = f"""Synthétise un menu hebdomadaire basé sur : {constraints}.
    Résultats des étapes : {all_work}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content

# FONCTION PRINCIPALE


@observe(name="plan_weekly_menu")
def plan_weekly_menu(constraints: str) -> dict:
    """Orchestration du menu"""
    
    get_client().update_current_trace(
        tags=["Groupe Baptiste_Clement", "Partie 3"],
        metadata={"constraints": constraints}
    )

    try:
        plan = _plan_steps(constraints)
        
        execution_results = []
        for i, step in enumerate(plan["steps"]):
            result = _execute_step(step, i, previous_results=execution_results)
            execution_results.append(result)
            
        final_menu = _synthesize_menu(constraints, execution_results)
        
        return {"status": "success", "menu": final_menu}

    except Exception as e:
        get_client().update_current_span(
            level="ERROR",
            status_message=f"Erreur globale : {str(e)}"
        )
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    contraintes = "Végétarien, budget 60€, focus légumes racines."
    print("ChefBot réfléchit...")
    res = plan_weekly_menu(contraintes)
    print(res.get("menu", res.get("message")))
    langfuse.flush()