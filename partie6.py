from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, WebSearchTool, VisitWebpageTool
from langfuse import observe, get_client
import litellm

load_dotenv()

model = LiteLLMModel(model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct")


def build_multi_agent_system():

    nutritionist_agent = CodeAgent(
        tools=[check_dietary_info],
        model=model,
        name="nutritionist_agent",
        description=(
            "A nutritionist agent that can provide expert advice on meal planning and dietary constraints. "
            "Give it a question about nutrition or meal planning and it will provide informed guidance."
        ),
        max_steps=5,
    )

    chef_agent = CodeAgent(
        tools=[check_fridge, get_recipe],
        model=model,
        name="chef_agent",
        description=(
            "A chef agent that can suggest recipes based on available ingredients and provide cooking instructions. " 
            "Give it a question about cooking or meal preparation and it will offer practical advice."
        ),
        max_steps=5,
    )

    budget_agent = CodeAgent(
        tools=[],
        model=model,
        name="budget_agent",
        description=(
            "A budget agent that can help optimize meal plans based on cost constraints. "
            "Give it a question about budgeting for meals and it will provide cost-effective suggestions."
        ),
        max_steps=5,
    )

    manager = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[nutritionist_agent, chef_agent, budget_agent],
        name="manager_agent",
        description=(
            "A manager agent that orchestrates the work of the nutritionist, chef, and budget agents to create a weekly meal plan. "
            "It takes dietary constraints as input and delegates tasks to the sub-agents to generate a comprehensive meal plan."
        ),
        max_steps=8,
    )
    return manager

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


# Create the system
manager_agent = build_multi_agent_system()

# Execute a query
query = query = """
Prépare un menu de 4 services pour 8 personnes (2 végé, 1 sans gluten, 1 sans arachides). 
Budget max: 120€. 
REMPLI : Demande au Chef les noms des plats et au Budget Agent le calcul final. 
Sois bref.
"""
response = manager_agent.run(query)

print(response)