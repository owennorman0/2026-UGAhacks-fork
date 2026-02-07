# agent.py
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
import json
import os

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
#collection = chroma_client.get_collection("dnd_knowledge")
claude = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

try:
    collection = chroma_client.get_collection("dnd_knowledge")
    print("Loaded existing vector store")
except:
    print("Building vector store...")
    collection = chroma_client.create_collection("dnd_knowledge")
    docs = []
    ids = []
    data_dir = os.path.join(os.path.dirname(__file__), "dnd_data")
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename)) as f:
                entries = json.load(f)
                for i, entry in enumerate(entries):
                    text = json.dumps(entry)
                    docs.append(text)
                    ids.append(f"{filename}_{i}")
    embeddings = embed_model.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, ids=ids)
    print(f"Indexed {len(docs)} documents")


SYSTEM_PROMPT = """You are a D&D Character Analyst. Given a description of a real person, 
you map them onto D&D 5e character attributes. Use the provided D&D reference data to justify 
your choices.

You MUST respond with ONLY valid JSON matching this exact structure. No markdown, no commentary, no explanation outside the JSON.

{
  "name": "<creative D&D character name inspired by the person>",
  "player": {
    "name": "<the person's real name if provided, otherwise 'Unknown'>"
  },
  "classes": [
    {
      "name": "<class name>",
      "level": 3,
      "hit_die": <hit die for class: Barbarian=12, Fighter/Paladin/Ranger=10, Bard/Cleric/Druid/Monk/Rogue/Warlock=8, Sorcerer/Wizard=6>
    }
  ],
  "race": {
    "name": "<D&D race that fits the person>",
    "size": "Medium",
    "speed": <race speed: Dwarf=25, most others=30>
  },
  "background": {
    "name": "<background name from SRD>",
    "feature": "<background feature name>"
  },
  "alignment": "<full alignment e.g. Lawful Good>",
  "experience_points": 900,
  "ability_scores": {
    "str": <4-20>,
    "dex": <4-20>,
    "con": <4-20>,
    "int": <4-20>,
    "wis": <4-20>,
    "cha": <4-20>
  },
  "armor_class": {
    "value": <total AC>,
    "base": 10,
    "armor": <armor bonus>,
    "shield": <shield bonus>
  },
  "speed": {
    "Walk": <base speed>,
    "Fly": 0,
    "Swim": 0,
    "Climb": 0,
    "Burrow": 0
  },
  "hit_points": {
    "max": <calculate from class hit die and CON modifier>,
    "current": <same as max>,
    "temp": 0
  },
  "hit_dice": {
    "total": "<e.g. 3d10>",
    "current": "3"
  },
  "death_saves": {
    "successes": 0,
    "failures": 0
  },
  "inspiration": false,
  "initiative_bonus": <DEX modifier>,
  "saving_throws": {
    "str": <true if class proficiency>,
    "dex": <true if class proficiency>,
    "con": <true if class proficiency>,
    "int": <true if class proficiency>,
    "wis": <true if class proficiency>,
    "cha": <true if class proficiency>
  },
  "skills": {
    "Acrobatics": false,
    "Animal Handling": false,
    "Arcana": false,
    "Athletics": false,
    "Deception": false,
    "History": false,
    "Insight": false,
    "Intimidation": false,
    "Investigation": false,
    "Medicine": false,
    "Nature": false,
    "Perception": false,
    "Performance": false,
    "Persuasion": false,
    "Religion": false,
    "Sleight of Hand": false,
    "Stealth": false,
    "Survival": false
  },
  "proficiencies": ["<list of weapon/armor/tool proficiencies>"],
  "languages": ["Common", "<other languages>"],
  "weapons": [
    {
      "name": "<weapon name>",
      "attack_bonus": <STR or DEX mod + proficiency>,
      "damage": "<damage dice + modifier>",
      "damage_type": "<slashing/piercing/bludgeoning>"
    }
  ],
  "currency": {
    "cp": 0,
    "sp": 0,
    "ep": 0,
    "gp": <reasonable starting gold>,
    "pp": 0
  },
  "equipment": ["<list of equipment items>"],
  "features_and_traits": ["<class and racial features>"],
  "feats": [],
  "personality": "<personality trait inspired by the person>",
  "ideal": "<ideal inspired by the person>",
  "bond": "<bond inspired by the person>",
  "flaw": "<flaw inspired by the person>",
  "details": {
    "personality": "<same as personality above>",
    "ideal": "<same as ideal above>",
    "bond": "<same as bond above>",
    "flaw": "<same as flaw above>"
  },
  "backstory": "<2-3 sentence D&D backstory inspired by the person's real traits>",
  "physical": {
    "age": <age if known, otherwise reasonable guess>,
    "height": "<height>",
    "weight": <weight>,
    "eyes": "<eye color>",
    "skin": "<skin description>",
    "hair": "<hair description>"
  },
  "allies_and_organizations": "<relevant faction or organization>",
  "treasure": "None of special value",
  "faction": {
    "name": "<organization name>",
    "rank": "<rank>",
    "contact": "<NPC contact name>"
  },
  "attacks_and_spellcasting": "<summary string of attacks>"
}

If the class is a spellcaster (Wizard, Sorcerer, Bard, Cleric, Druid, Warlock, Paladin, Ranger), also include:
"spellcasting": {
  "class": "<class>",
  "ability": "<spellcasting ability>",
  "spell_save_dc": <8 + proficiency + ability mod>,
  "spell_attack_bonus": <proficiency + ability mod>,
  "spell_slots": { "level_1": { "total": <slots>, "remaining": <slots> }, "level_2": { "total": <slots>, "remaining": <slots> } },
  "cantrips_known": [ { "name": "<cantrip>", "level": 0 } ],
  "spells_known": [ { "name": "<spell>", "level": 1, "prepared": true } ]
}

Set skill proficiencies to true based on class and background. Choose 2-4 skills that make sense.
Calculate all numbers correctly using D&D 5e rules.
Use the D&D reference data to justify your class, alignment, and trait choices - but only output JSON, no explanations."""


"""SYSTEM_PROMPT = You are a D&D Character Analyst. Given a description of a real person, 
you map them onto D&D 5e character attributes. Use the provided D&D reference data to justify 
your choices. Return a structured character sheet including:

- Class (and subclass if clear)
- Alignment
- Ability Scores (relative ranking, not exact numbers)
- Background
- 2 Personality Traits, 1 Ideal, 1 Bond, 1 Flaw
- Brief narrative justification

Be specific and cite which traits/behaviors map to which D&D elements."""


def analyze_person(description: str) -> str:
    # Retrieve relevant D&D context
    #print("Retrieving D&D context...")
    query_embedding = embed_model.encode([description]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)
    context = "\n\n".join(results["documents"][0])

    #print("calling claude api, might take time")
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"""## D&D Reference Data:
{context}

## Person Description:
{description}

Analyze this person and generate their D&D character sheet."""
        }]
    )
    #print("done")
    #print(response)
    return response.content[0].text