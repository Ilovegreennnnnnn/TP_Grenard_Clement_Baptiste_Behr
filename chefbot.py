import os
from typing import Any
from dotenv import load_dotenv
from groq import Groq
from litellm import api_key
import litellm
from smolagents import CodeAgent, LiteLLMModel
from langfuse import get_client, observe
import json
from smolagents import tool

load_dotenv()
langfuse_client = get_client() 
groq_api_key = os.getenv("GROQ_API_KEY")

model = LiteLLMModel(
    model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key
)

groq_client = Groq(api_key=groq_api_key)

#Partie 1 :

@observe(name="ChefBot")
def ask_chef(question: str) -> str:

    langfuse_client.update_current_trace(
        metadata={
            "type": "Groupe Clement et Baptiste, Partie 1",
            "season": "Hiver"
        },
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system", 
                "content": (
                    "Tu es ChefBot, un grand chef cuisinier français spécialisé en cuisine de saison. "
                    "Ton expertise porte sur les produits frais du terroir. "
                    "Réponds avec élégance et conseille toujours des ingrédients de saison."
                )
            },
            {"role": "user", "content": question}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content

def ask_chef_partie_1():
    for i in [0.1, 0.7, 1.2]:
        print(f"\n--- Question avec température {i} ---")
        reponse = ask_chef("Quel dessert léger puis-je préparer en hiver ?")
        print(f"ChefBot: {reponse}")
    langfuse_client.flush()

#ask_chef_partie_1()

#À 0.1 : Vous avez eu une recette de tarte très détaillée, presque une fiche technique de cuisine.

#À 0.7 : Le modèle a commencé à mélanger les fruits (pommes et poires) 
# et à suggérer des accompagnements comme la crème anglaise.

#À 1.2 : Il ne s'est plus arrêté à une recette, il vous a balayé tout un éventail de possibilités, 
# du sorbet au mille-feuille, en étant beaucoup plus suggestif.


#Partie 2 :

def call_groq(system_prompt: str, user_content: str, json_mode: bool = False) -> str:
    """Helper pour centraliser les appels Groq"""
    response_format = {"type": "json_object"} if json_mode else {"type": "text"}
    
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.4,
        response_format=response_format
    )
    return completion.choices[0].message.content

@observe(name="plan_weekly_menu")
def plan_weekly_menu(constraints: str) -> dict:
    """Fonction principale avec gestion d'erreur globale."""
    
    langfuse_client.update_current_trace(
        metadata={"agent": "ChefBot-MultiStep"},
        tags=["Groupe Clement et Baptiste", "Partie 2"]
    )

    try:
        plan = planification_menu(constraints)

        step_results = []
        for i, step in enumerate(plan["steps"]):
            result = execution_menu(step, i, context=step_results)
            step_results.append(result)

        final_menu = synthese_menu(constraints, step_results)

        return {
            "status": "success",
            "final_menu": final_menu
        }

    except Exception as e:
        langfuse_client.update_current_trace(
            level="ERROR",
            status_message=f"Erreur critique dans le pipeline : {str(e)}"
        )
        return {"status": "error", "error": str(e)}

@observe(name="planification_menu", as_type="generation")
def planification_menu(constraints: str, retry: bool = False) -> dict:
    system_prompt = (
        "Tu es l'assistant de planification de ChefBot. Décompose la création d'un menu complet "
        "(Petit-déjeuner, Déjeuner, Dîner) en 3 étapes logiques. "
        "Réponds UNIQUEMENT en JSON."
    )
    user_content = f"Contraintes : {constraints}. \nFormat: {{'steps': ['étape 1', 'étape 2', ...], 'reasoning': '...'}}"
    
    response_text = call_groq(system_prompt, user_content, json_mode=True)
    
    try:
        plan = json.loads(response_text)
        if "steps" not in plan:
            raise ValueError("Clé 'steps' manquante dans le JSON")
        
        return plan

    except (json.JSONDecodeError, ValueError) as e:
        if not retry:
            print(f"Erreur JSON détectée. Tentative de Retry...")
            return planification_menu(constraints, retry=True)
        else:
            langfuse_client.update_current_span(
                level="ERROR",
                status_message=f"Échec définitif du parsing JSON après retry : {str(e)}",
                metadata={"raw_response": response_text}
            )
            raise e

@observe(name="execution_menu")
def execution_menu(step: str, index: int, context: list) -> dict:
    context_str = "\n".join([f"- {r['output']}" for r in context]) if context else "Aucun historique."
    
    system_prompt = (
        "Tu es ChefBot, expert en cuisine de saison. Exécute précisément l'étape demandée "
        "en tenant compte du travail déjà effectué précédemment."
    )
    user_content = f"Historique du menu :\n{context_str}\n\nÉtape à réaliser : {step}"
    
    output = call_groq(system_prompt, user_content)
    
    return {"step": step, "index": index, "output": output}

@observe(name="synthese-phase", as_type="generation")
def synthese_menu(constraints: str, results: list) -> str:
    results_text = "\n\n".join([f"Phase {r['index']+1} ({r['step']}) : {r['output']}" for r in results])
    
    system_prompt = (
        "Tu es ChefBot. Compile les réflexions précédentes en un menu hebdomadaire structuré, "
        "élégant et respectant les contraintes de saison."
    )
    user_content = f"Contraintes : {constraints}\n\nTravail préparatoire :\n{results_text}"
    
    return call_groq(system_prompt, user_content)


def partie_2():
    print("\n--- Planificateur ChefBot Hebdomadaire ---")
    ma_demande = "Famille de 4 personnes, budget serré, ingrédients de février uniquement."
    
    final_output = plan_weekly_menu(ma_demande)
    
    if final_output["status"] == "success":
        print("\n=== MENU FINAL GÉNÉRÉ ===")
        print(final_output["final_menu"])
    else:
        print(f"Erreur : {final_output['error']}")
    
    langfuse_client.flush()


#parite_2()

#Partie 4 :

FRIDGE_CONTENT = ["oeufs", "tomates", "fromage", "lait", "basilic"]

RECIPES = {
    "omelette": "Battre les oeufs, ajouter le fromage, cuire à la poêle.",
    "salade caprese": "Couper tomates et mozzarella, ajouter basilic et huile d'olive."
}

DIETARY_INFO = {
    "oeufs": {"calories": 70, "allergenes": ["oeuf"]},
    "fromage": {"calories": 110, "allergenes": ["lait"]},
    "tomates": {"calories": 20, "allergenes": []}
}

# Outils

@tool
def check_fridge():
    """
    Retourne une liste d'ingrédients disponibles dans le frigo.
    """
    return ", ".join(FRIDGE_CONTENT)

@tool
def get_recipe(dish_name: str) -> str:
    """
    Retourne une recette détaillée pour un plat donné.
    Args:
        dish_name: Le nom du plat recherché.
    """
    # Recherche simple
    for name, recipe in RECIPES.items():
        if dish_name.lower() in name:
            return recipe
    return "Recette non trouvée."

@tool
def check_dietary_info(ingredient: str) -> str:
    """
    Retourne les informations nutritionnelles et allergéniques d'un ingrédient.
    Args:
        ingredient: Le nom de l'ingrédient.
    """
    info = DIETARY_INFO.get(ingredient.lower())
    if info:
        return f"Infos pour {ingredient}: {info}"
    return f"Pas d'infos pour {ingredient}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_fridge",
            "description": "Retourne une liste d'ingrédients disponibles dans le frigo",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recipe",
            "description": "Retourne une recette pour un plat",
            "parameters": {
                "type": "object",
                "properties": {
                    "dish_name": {"type": "string", "description": "Nom du plat"}
                },
                "required": ["dish_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_dietary_info",
            "description": "Donne les infos nutritionnelles d'un ingrédient",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredient": {"type": "string", "description": "Nom de l'ingrédient"}
                },
                "required": ["ingredient"]
            }
        }
    }
]

TOOL_REGISTRY = {
    "check_fridge": check_fridge,
    "get_recipe": get_recipe,
    "check_dietary_info": check_dietary_info
}

def tool_calling_agent(user_message: str) -> str:
    """
    A simple tool-calling loop:
    1. Send user message + tool definitions to the LLM
    2. If the LLM wants to call tools, execute them
    3. Send tool results back to the LLM
    4. Repeat until the LLM gives a final text response
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided tools when needed to answer questions accurately."
        },
        {"role": "user", "content": user_message}
    ]

    for iteration in range(5):  
        print(f"\n  [Iteration {iteration + 1}]")

        response = groq_client.chat.completions.create(
            model="groq/llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            print(f"  Final answer ready.")
            return message.content

        messages.append(message) 

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"  Tool call: {name}({args})")

            # Execute the tool
            func = TOOL_REGISTRY.get(name)
            if func:
                result = func(**args)
            else:
                result = f"Error: unknown tool '{name}'"

            print(f"  Result: {result}")

            # Add tool result to message history
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Error: max iterations reached"


def partie_4_manuel():
    print("=" * 60)
    print("CHEFBOT: TOOL USE / FUNCTION CALLING DEMO")
    print("=" * 60)

    # Example 1: Simple tool call (Vérification du stock)
    print("\n--- Example 1: Inventory Check ---")
    answer = tool_calling_agent("Quels sont les ingrédients disponibles dans le frigo ?")
    print(f"\nAnswer: {answer}")

    # Example 2: Multiple tool calls (Frigo + Recette + Diététique)
    print("\n--- Example 2: Complex Chain (Fridge + Recipe) ---")
    answer = tool_calling_agent(
        "Regarde ce qu'il y a dans le frigo. Est-ce que je peux faire une salade caprese ? "
        "Si oui, donne-moi la recette et dis-moi si le fromage est calorique."
    )
    print(f"\nAnswer: {answer}")

    # Example 3: Specific Query (Dietary Info)
    print("\n--- Example 3: Specific Tool (Dietary Info) ---")
    answer = tool_calling_agent("Quelles sont les informations nutritionnelles pour les oeufs ?")
    print(f"\nAnswer: {answer}")

#partie_4_manuel()

agent = CodeAgent(
    tools=[check_fridge, get_recipe, check_dietary_info], 
    model=model,
    add_base_tools=False,
    max_steps=5
)

def partie_4_smolagent():
    print("=" * 60)
    print("CHEFBOT: TOOL USE / FUNCTION CALLING DEMO")
    print("=" * 60)

    # Example 1: Simple tool call (Vérification du stock)
    print("\n--- Example 1: Inventory Check ---")
    answer = agent.run("Quels sont les ingrédients disponibles dans le frigo ?")
    print(f"\nAnswer: {answer}")

    # Example 2: Multiple tool calls (Frigo + Recette + Diététique)
    print("\n--- Example 2: Complex Chain (Fridge + Recipe) ---")
    answer = agent.run(
        "Regarde ce qu'il y a dans le frigo. Est-ce que je peux faire une salade caprese ? "
        "Si oui, donne-moi la recette et dis-moi si le fromage est calorique."
    )
    print(f"\nAnswer: {answer}")

    # Example 3: Specific Query (Dietary Info)
    print("\n--- Example 3: Specific Tool (Dietary Info) ---")
    answer = agent.run("Quelles sont les informations nutritionnelles pour les oeufs ?")
    print(f"\nAnswer: {answer}")

partie_4_smolagent()

#Partie 6