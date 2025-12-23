import run_vlm
import run_llm
from pdf2image import convert_from_path
import os
import tqdm
import json
import run_embedding
from math import sqrt

import networkx as nx
import community as community_louvain

# Load prompts
with open("entitiesprompt.txt", "r") as f:
    entitiesprompt = f.read()

with open("relationshipsprompt.txt", "r") as f:
    relationshipsprompt = f.read()

#region Functions to find entities and relationships
def find_entities(image):
    valid_entities = False
    while not valid_entities:
        output = run_vlm.run_prompt(image, entitiesprompt).replace("```json","").replace("```","")
        try:
            output_json = json.loads(output)
            entities = output_json["entities"]
            valid_entities = True
        except:
            print("Bad boy, bad json")
            print(output)
    return entities

def find_relationships(image, entities):
    valid_relationships = False
    while not valid_relationships:
        output = run_vlm.run_prompt(image, relationshipsprompt + str(entities)).replace("```json","").replace("```","")
        try:
            output_json = json.loads(output)
            relationships = output_json["relationships"]
            valid_relationships = True
        except:
            print("Bad boy, bad json")
            print(output)
    return relationships
#endregion

# Loop through all PDFs
for file in os.listdir("data\documents"):
    if not ".pdf" in file:
        pass
    
    images = convert_from_path("data\documents\\" + file)

    #region Create entities and relationships
    totalentities = []
    totalrelationships = []
    for image in tqdm.tqdm(images, desc=f"Entities and relationships extraction"):
        image.show()
        entities=find_entities(image)
        for entity in entities:
            totalentities.append(entity)

        relationships = find_relationships(image, entities)
        for relationship in relationships:
            totalrelationships.append(relationship)
    #endregion

    #region Merge entities
    final_entities = []

    saved_texts = []

    for entity in tqdm.tqdm(totalentities, "Merging entities"):
        # create a text representation for the entity to compare
        if isinstance(entity, dict):
            text = entity.get("name") or entity.get("text") or json.dumps(entity, ensure_ascii=False)
        else:
            text = str(entity)

        is_duplicate = False
        for s_text in saved_texts:
            sim = run_embedding.calculate_cosine_similarity(text, s_text)
            if sim > 0.90:
                is_duplicate = True
                break

        if not is_duplicate:
            final_entities.append(entity)
            saved_texts.append(text)

    # build text representations for finalized entities
    final_entity_texts = []
    for ent in tqdm.tqdm(final_entities, "Preparing entity texts"):
        if isinstance(ent, dict):
            final_entity_texts.append(ent.get("name") or ent.get("text") or json.dumps(ent, ensure_ascii=False))
        else:
            final_entity_texts.append(str(ent))
    #endregion

    #region Connect relationships to entities
    final_relationships = []

    for rel in tqdm.tqdm(totalrelationships, desc="Assigning relationship entity indices"):
        # ensure relationship is a dict copy so we don't mutate originals
        rel_obj = rel.copy() if isinstance(rel, dict) else {"subject": str(rel)}

        subj_text = str(rel_obj.get("subject", ""))
        obj_text = str(rel_obj.get("object", ""))

        # defaults if no entities available
        best_subj_idx, best_obj_idx = -1, -1
        best_subj_sim, best_obj_sim = 0.0, 0.0

        for idx, ent_text in enumerate(final_entity_texts):
            try:
                sim_subj = run_embedding.calculate_cosine_similarity(subj_text, ent_text)
            except Exception:
                sim_subj = 0.0
            try:
                sim_obj = run_embedding.calculate_cosine_similarity(obj_text, ent_text)
            except Exception:
                sim_obj = 0.0

            if sim_subj > best_subj_sim:
                best_subj_sim = sim_subj
                best_subj_idx = idx
            if sim_obj > best_obj_sim:
                best_obj_sim = sim_obj
                best_obj_idx = idx

        # store indices (and optionally similarities) on the relationship
        rel_obj["subject_index"] = best_subj_idx
        rel_obj["object_index"] = best_obj_idx

        final_relationships.append(rel_obj)
    #endregion

    #region Cluster into communities using the Louvain algorithm
    clusters = []

    # Build a graph where nodes are final entities and edges are relationships
    if len(final_entities) > 0:
        G = nx.Graph()

        # add entity nodes
        for idx, ent in enumerate(final_entities):
            G.add_node(idx, entity=ent)

        # add edges from relationships when subject/object indices exist
        for rel in final_relationships:
            si = rel.get("subject_index", -1)
            oi = rel.get("object_index", -1)
            # validate indices
            if si is None or oi is None:
                continue
            try:
                si_i = int(si)
                oi_i = int(oi)
            except Exception:
                continue

            if si_i >= 0 and oi_i >= 0 and si_i < len(final_entities) and oi_i < len(final_entities) and si_i != oi_i:
                weight = rel.get("weight", 1)
                if G.has_edge(si_i, oi_i):
                    # accumulate weight if edge already exists
                    G[si_i][oi_i]["weight"] = G[si_i][oi_i].get("weight", 1) + weight
                else:
                    G.add_edge(si_i, oi_i, weight=weight)

        # If no edges, each node becomes its own cluster
        if G.number_of_nodes() == 0:
            clusters = []
        else:
            try:
                partition = community_louvain.best_partition(G, weight='weight')
                # partition: node -> community_id
                comm_map = {}
                for node, comm in partition.items():
                    comm_map.setdefault(comm, []).append(node)

                clusters = []
                for comm_id, nodes in comm_map.items():
                    clusters.append({
                        "community_id": comm_id,
                        "indices": sorted(nodes),
                        "entities": [final_entities[i] for i in sorted(nodes)]
                    })
            except Exception:
                # fallback to connected components
                clusters = []
                for i, comp in enumerate(nx.connected_components(G)):
                    nodes = sorted(list(comp))
                    clusters.append({
                        "community_id": i,
                        "indices": nodes,
                        "entities": [final_entities[n] for n in nodes]
                    })
    else:
        # no networkx available or no entities found: leave clusters empty
        clusters = []

    print("Found", len(clusters), "clusters")
    #endregion

    #region Generate summaries for each cluster
    for cluster in tqdm.tqdm(clusters, desc="Generating cluster summaries"):
        entity_texts = [str(ent) for ent in cluster["entities"]]
        prompt = "Summarize the following entities into a concise description:\n\n" + "\n".join(entity_texts)
        summary = run_llm.run_prompt(prompt)
        cluster["summary"] = summary
    #endregion
    print(clusters)