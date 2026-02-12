#Pour cette partie, on a testé de nombreux modèles mais à chaque fois, nous avons eu des problèmes de token/rate limit. 
#Nous avons finalement opté pour le modèle "llama-4-scout-17b-16e-instruct" de Groq, qui est plus léger et nous a permis de faire tourner l'agent sans problème.
#Cependant le cycle ne finit pas car il y a aussi une erreur de ratelimit sur ce modèle...



import os
import litellm
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, Tool
from langfuse import get_client, observe

load_dotenv()

litellm.callbacks = ["langfuse_otel"]

model = LiteLLMModel(model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct")

# 5.1 - OUTIL DE BASE DE DONNÉES 

class MenuDatabaseTool(Tool):
    """
    Outil de recherche dans la base de données du restaurant.
    Hérite de la classe Tool pour un contrôle précis des entrées.
    """
    name = "menu_search"
    description = "Recherche des plats selon la catégorie, le prix max et les allergènes."
    inputs = {
        "prix_max": {"type": "number", "description": "Budget maximum par plat.", "nullable": True},
        "categorie": {"type": "string", "description": "entrée, plat ou dessert.", "nullable": True},
        "allergene_absent": {"type": "string", "description": "Allergène à exclure (ex: 'gluten').", "nullable": True},
        "vegetarien": {"type": "boolean", "description": "Filtrer les plats végétariens.", "nullable": True}
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.menu = [
            {"nom": "Salade César", "prix": 12, "cat": "entrée", "vege": False, "allergenes": ["gluten"]},
            {"nom": "Velouté de Potiron", "prix": 9, "cat": "entrée", "vege": True, "allergenes": []},
            {"nom": "Risotto aux Cèpes", "prix": 18, "cat": "plat", "vege": True, "allergenes": []},
            {"nom": "Burger Sans Gluten", "prix": 20, "cat": "plat", "vege": False, "allergenes": []},
            {"nom": "Ratatouille", "prix": 16, "cat": "plat", "vege": True, "allergenes": []},
            {"nom": "Steak Frites", "prix": 22, "cat": "plat", "vege": False, "allergenes": []},
            {"nom": "Pâtes au Pesto", "prix": 15, "cat": "plat", "vege": True, "allergenes": ["gluten"]},
            {"nom": "Mousse Chocolat", "prix": 8, "cat": "dessert", "vege": True, "allergenes": []},
            {"nom": "Salade de Fruits", "prix": 7, "cat": "dessert", "vege": True, "allergenes": []},
            {"nom": "Tarte Tatin", "prix": 9, "cat": "dessert", "vege": True, "allergenes": ["gluten"]}
        ]

    def forward(self, prix_max: float = None, categorie: str = None, allergene_absent: str = None, vegetarien: bool = False) -> str:
        results = self.menu
        if prix_max:
            results = [p for p in results if p["prix"] <= prix_max]
        if categorie:
            results = [p for p in results if p["cat"] == categorie.lower()]
        if vegetarien:
            results = [p for p in results if p["vege"]]
        if allergene_absent:
            results = [p for p in results if allergene_absent.lower() not in [a.lower() for a in p["allergenes"]]]
        
        if not results:
            return "Aucun plat ne correspond à vos critères."
        return "\n".join([f"- {p['nom']} ({p['prix']}€)" for p in results])

@tool
def calculate(expression: str) -> str:
    """
    Évalue une expression mathématique pour calculer le total.

    Args:
        expression: L'opération à effectuer (ex: '18 + 20 + 15').
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Erreur: caractères non autorisés."
        return str(eval(expression))
    except Exception as e:
        return f"Erreur de calcul: {e}"

# 5.2 - AGENT AVEC PLANIFICATION 

instructions = (
    "Tu es un serveur de restaurant expert. Tu agis en tant que CodeAgent.\n"
    "IMPORTANT - FORMAT DE RÉPONSE :\n"
    "1. Tu dois TOUJOURS écrire ton code Python entre les balises <code> et </code>.\n"
    "2. NE PAS UTILISER de blocs Markdown avec trois backticks (```python).\n"
    "3. Ton code doit être valide et utiliser les outils mis à ta disposition.\n"
    "\n"
    "DÉMARCHE :\n"
    "- Cherche les plats via 'menu_search'.\n"
    "- Calcule le total avec 'calculate'.\n"
    "- Respecte les contraintes (Végétarien, Sans Gluten, Budget 60€).\n"
    "- Utilise 'final_answer' pour conclure avec le menu complet."
)

agent = CodeAgent(
    tools=[MenuDatabaseTool(), calculate],
    model=model,
    planning_interval=2,
    max_steps=8, 
    instructions=instructions
)

@observe(name="restaurant-demo")
def run_restaurant():
    print("--- 5.2 : Test Agent Planificateur ---")
    query = "On est 3. Un végétarien, un sans gluten, et moi je mange de tout. Budget max 60€ au total. Proposez-nous un menu."
    print(agent.run(query))

    print("\n--- 5.3 : Dialogue Multi-tours ---")
    
    print("User: Finalement, un autre dessert (sans gluten) pour tout le groupe.")
    res2 = agent.run("Propose un autre dessert sans gluten à la place. Recalcule le total.", reset=False)
    print(f"Agent: {res2}")

    print("\nUser: C'est parfait, l'addition s'il vous plaît.")
    res3 = agent.run("Donne-nous l'addition finale détaillée.", reset=False)
    print(f"Agent: {res3}")

if __name__ == "__main__":
    run_restaurant()
    get_client().flush()