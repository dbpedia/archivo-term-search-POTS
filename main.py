from create_indexes import createIndexes
from langchain_community.embeddings import SentenceTransformerEmbeddings
from nlp.parsers import *
from configs import *
from dotenv import load_dotenv
from nlp.normalizer import *
from sparql.Endpoint import Endpoint
from index.import_index import *
import os
import csv
import time
from SPARQLWrapper import SPARQLWrapper, JSON
#OpenAI
load_dotenv()

# Filter for class / property
# Understand text2kg code +-
# Try removing terms from ontologies to see if scores change

url_endpoint = "http://95.217.207.179:8995/sparql/"
def get_ontology_names(url_endpoint):
    query = """SELECT DISTINCT ?graph
    WHERE {
    GRAPH ?graph {
        ?s ?p ?o.
    }
    }
    """
   
    sparql = SPARQLWrapper(url_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    #print(results["results"])
    all_ontologies = []
    for r in results["results"]["bindings"]:
        
        full = r["graph"]["value"].split("/")[-1]
        if ".ttl" in full and ("dbpedia" in full or "wikidata" in full):
            full = full.replace(".ttl", "")
            if "_number=" in full:
                full = "_".join(full.split("_number=")[:-1])
            full = full.split("_")[1].replace("domain=", "")
            all_ontologies.append(full)
    return all_ontologies
#print(results)

# all_ontologies = ["ont_3_airport",
# "ont_17_artist",
# "ont_9_astronaut",
# "ont_5_athlete",
# "ont_4_book",
# "ont_4_building",
# "ont_8_celestialbody",
# "ont_16_city",
# "ont_10_comicscharacter",
# "ont_7_company",
# "ont_6_computer",
# "ont_10_culture",
# "ont_19_film",
# "ont_13_food",
# "ont_11_meanoftransportation",
# "ont_5_military",
# "ont_12_monument",
# "ont_1_movie",
# "ont_2_musicalwork",
# "ont_9_nature",
# "ont_6_politician",
# "ont_8_politics",
# "ont_18_scientist",
# "ont_7_space",
# "ont_3_sport",
# "ont_15_sportsteam",
# "ont_1_university",
# "ont_14_writtenwork"]
all_ontologies = get_ontology_names(url_endpoint)
model_list = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]
with open("test_qa.txt", "w") as f:
    pass   
with open("logs.json", "w") as f:
    pass    
with open("overall_results.csv", "w") as f:
    pass
f.close()

document_to_interpret = "example.pdf"

fetchable_datatypes = ["properties","classes"]

# Reads a text file (PDF, CSV, ETC)
#if document_to_interpret.endswith(".pdf"):

import json

def load_text_and_triples(ontology_name):
    # Define the path to the .jsonl file
    file_path = f'./text2kg_benchmarks/{ontology_name}_train.jsonl'

    # Initialize an empty list to store the JSON objects
    data = []

    # Open the .jsonl file and read line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object
            json_object = json.loads(line.strip())
            # Add the JSON object to the list
            data.append(json_object)
    return data

methodologies = ["Term"]#,"Term+Description","Term+Subclasses","Term+Superclasses","Term+SeeAlso"] #,"Term+Definition"]

default_failed = set()

# Searches the T-box index for matches for each term
model_scores = {method: {k: [] for k in model_list} for method in methodologies}

model_scores = {}
methodology_scores = {}



for search_type in fetchable_datatypes:
    model_scores[search_type] = {}
    methodology_scores[search_type] = {}
    
    with open("test_qa.csv", "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([f"--- Search type: {search_type} ---"])

    for ontology in all_ontologies:
        try:
            benchmark_data = load_text_and_triples(ontology)
            
            # Initialize scores for this ontology
            model_scores[search_type][ontology] = {}
            methodology_scores[search_type][ontology] = {}
            
            # Initialize each methodology score container
            for methodology in methodologies:
                methodology_scores[search_type][ontology][methodology] = []
            
            # Open CSV file to start logging
            with open("test_qa.csv", "a", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([f"Ontology: {ontology}"])
            
            # For each question in metadata
            for case in benchmark_data[:5]:
                text = case["sent"]
                triples = case["triples"]
                all_terms = set()
                for x in triples:
                    all_terms.add(x["rel"])
                    
                every_triple_mentioned = set([x.values() for x in triples])
                
                expected_output = [x.lower() for x in all_terms]
                
                with open("test_qa.csv", "a", newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([f"Question: {text}"])
                    csvwriter.writerow([f"Expected: {expected_output}"])
                
                # For each methodology
                for methodology in methodologies:
                    with open("test_qa.csv", "a", newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow([f"Method: {methodology}"])
                    
                    # Initialize scores for this methodology
                    if methodology not in model_scores[search_type][ontology]:
                        model_scores[search_type][ontology][methodology] = {}
                    
                    # For each model
                    for model in model_list:
                        with open("test_qa.csv", "a", newline='') as csvfile:
                            csvwriter = csv.writer(csvfile)
                            csvwriter.writerow([f"Model: {model}"])
                        
                        # Create indexes and visualize
                        embedding_function = SentenceTransformerEmbeddings(model_name=model)
                        endpoint_t_box, t_box_index, normalizer = createIndexes(embedding_function, methodology, ontology, search_type)
                        
                        if not os.path.exists(f"./pca_results/{ontology}_{search_type}"):
                            os.mkdir(f"./pca_results/{ontology}_{search_type}")
                        
                        t_box_index.visualize_index(method='pca', n_components=2, label_size=10, output_file=f'./pca_results/{ontology}_{search_type}/tbox_index_{methodology}_{model}.png')
                        
                        # Parse text and compute result
                        result = parseText(text, t_box_index, normalizer, endpoint_t_box)
                        result_labels = set([x["label"].lower() for x in result])
                        formatted_outcome = {"extra": [], "missed": []}
                        
                        not_found = [x for x in expected_output if x not in result_labels]
                        too_much = [x for x in result_labels if x not in expected_output]
                        
                        not_found_str = " failed to find "+str(not_found) if len(not_found) > 0 else ""
                        too_much_str = " found extra "+str(too_much) if len(too_much) > 0 else ""
                        extra = " and".join(filter(None, [not_found_str, too_much_str]))
                        
                        hits = (len([x for x in result_labels if x in expected_output])) / len(expected_output) * 100
                        missing = len(not_found) / len(expected_output) * 100
                        score = hits - (len(too_much)) / len(expected_output) * 100
                        
                        # Track scores
                        if model not in model_scores[search_type][ontology][methodology]:
                            model_scores[search_type][ontology][methodology][model] = []
                        
                        model_scores[search_type][ontology][methodology][model].append(score)
                        methodology_scores[search_type][ontology][methodology].append(score)
                        
                        with open("test_qa.csv", "a", newline='') as csvfile:
                            csvwriter = csv.writer(csvfile)
                            csvwriter.writerow([
                                f"Hits: {hits}%",
                                f"Missing: {not_found}",
                                f"Extras: {too_much}",
                                f"Score: {score}"
                            ])
                            csvwriter.writerow([
                                ""
                            ])
            
            # Log average scores for each model per ontology
            with open("overall_results.csv", "a", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                
                # Write header for the ontology
                csvwriter.writerow([f"Ontology {ontology}"])
                
                # Write model scores
                for methodology, models in model_scores[search_type][ontology].items():
                    for model, scores in models.items():
                        avg_score = sum(scores) / len(scores) if scores else 0
                        csvwriter.writerow([f"Model: {model}", f"Methodology: {methodology}", f"Average Score: {avg_score}"])
                
                # Write average score for each methodology per ontology
                for methodology, scores in methodology_scores[search_type][ontology].items():
                    avg_methodology_score = sum(scores) / len(scores) if scores else 0
                    csvwriter.writerow([f"Average Score for Methodology {methodology}", f"Average Score: {avg_methodology_score}"])
        except Exception as e:
            import traceback
            with open("errors.txt", "a") as f:
                f.write("Error on "+str(ontology)+"\n"+str(e)+"\n")
                traceback.print_exc(file=f)
# model_scores = {}
# methodology_scores = {}

# # For each ontology
# for ontology in all_ontologies:
#     try:
#         benchmark_data = load_text_and_triples(ontology)
        
#         # Initialize scores for this ontology
#         model_scores[search_type][ontology] = {}
#         methodology_scores[search_type][ontology] = {}
        
#         # Initialize each methodology score container
#         for methodology in methodologies:
#             methodology_scores[search_type][ontology][methodology] = []
        
#         # Open file to start logging
#         with open("test_qa.txt", "a") as f:
#             f.write(f"--------------------------------------- Ontology: {ontology} ---------------------------------------\n")
        
#         # For each question in metadata
#         for case in benchmark_data:
#             text = case["sent"]
#             triples = case["triples"]
#             all_terms = set()
#             for x in triples:
#                 #all_terms.add(x["sub"])
#                 all_terms.add(x["rel"])
#                 #all_terms.add(x["sub"])
                
#             every_triple_mentioned = set([x.values() for x in triples])
            
#             expected_output = [x.lower() for x in all_terms]
            
#             with open("test_qa.txt", "a") as f:
#                 f.write(f"--------------------------------------- Question: {text} ---------------------------------------\n")
#                 f.write(f"--------------------------------------- Expected: {expected_output} ---------------------------------------\n")
            
#             # For each methodology
#             for methodology in methodologies:
#                 with open("test_qa.txt", "a") as f:
#                     f.write(f"--------------------------------------- Method: {methodology} ---------------------------------------\n".replace("-", "*"))
                
#                 # Initialize scores for this methodology
#                 if methodology not in model_scores[search_type][ontology]:
#                     model_scores[search_type][ontology][methodology] = {}
                
#                 # For each model
#                 for model in model_list:
#                     with open("test_qa.txt", "a") as f:
#                         f.write(f"\nModel: {model}\n")
                    
#                     # Create indexes and visualize
#                     embedding_function = SentenceTransformerEmbeddings(model_name=model)
#                     endpoint_t_box, t_box_index, normalizer = createIndexes(embedding_function, methodology, ontology)
                    
#                     if not os.path.exists(f"./pca_results/{ontology}"):
#                         os.mkdir(f"./pca_results/{ontology}")
                    
#                     t_box_index.visualize_index(method='pca', n_components=2, label_size=10, output_file=f'./pca_results/{ontology}/tbox_index_{methodology}_{model}.png')
                    
#                     # Parse text and compute result
#                     result = parseText(text, t_box_index, normalizer, endpoint_t_box)
#                     result_labels = set([x["label"].lower() for x in result])
#                     formatted_outcome = {"extra": [], "missed": []}
                    
#                     not_found = [x for x in expected_output if x not in result_labels]
#                     too_much = [x for x in result_labels if x not in expected_output]
                    
#                     not_found_str = " failed to find "+str(not_found) if len(not_found) > 0 else ""
#                     too_much_str = " found extra "+str(too_much) if len(too_much) > 0 else ""
#                     extra = " and".join(filter(None, [not_found_str, too_much_str]))
                    
#                     hits = (len([x for x in result_labels if x in expected_output])) / len(expected_output) * 100
#                     missing = len(not_found) / len(expected_output) * 100
#                     score = hits - (len(too_much)) / len(expected_output) * 100
                    
#                     # Track scores
#                     if model not in model_scores[search_type][ontology][methodology]:
#                         model_scores[search_type][ontology][methodology][model] = []
                    
#                     model_scores[search_type][ontology][methodology][model].append(score)
#                     methodology_scores[search_type][ontology][methodology].append(score)
                    
#                     with open("test_qa.txt", "a") as f:
#                         f.write(f"\nHits: {hits}%\n")
#                         f.write(f"Missing: {not_found}\n")
#                         f.write(f"Extras: {too_much}\n")
#                         f.write(f"**Score**: {score}\n")
        
#         # Log average scores for each model per ontology
#         with open("overall_results.csv", "a", newline='') as csvfile:
#             csvwriter = csv.writer(csvfile)
            
#             # Write header for the ontology
#             csvwriter.writerow([f"Ontology {ontology}"])
            
#             # Write model scores
#             for methodology, models in model_scores[search_type][ontology].items():
#                 for model, scores in models.items():
#                     avg_score = sum(scores) / len(scores) if scores else 0
#                     csvwriter.writerow([f"Model: {model}", f"Methodology: {methodology}", f"Average Score: {avg_score}"])
            
#             # Write average score for each methodology per ontology
#             for methodology, scores in methodology_scores[search_type][ontology].items():
#                 avg_methodology_score = sum(scores) / len(scores) if scores else 0
#                 csvwriter.writerow([f"Average Score for Methodology {methodology}", f"Average Score: {avg_methodology_score}"])
#     except Exception as e:
#         import traceback
#         with open("errors.txt", "a") as f:
#             f.write("Error on"+str(ontology)+" "+str(e))
#             traceback.print_exc()
#         f.close()

# for method in model_scores:
#     print("METHOD:", method)
#     for model in model_scores[method]:
#         print(model, ":", sum(model_scores[method][model])/len(model_scores[method][model]))
                
                


# texts_and_expected_matches = {content[k]: k for k in content}
# default_failed = set()
# for text in texts_and_expected_matches:
#     expected_output = set(texts_and_expected_matches[text])
#     # Parses the contents within that file
#     #print(expected_output)
#     result = parseText(text, t_box_index, normalizer, endpoint_t_box)
#     result_labels = set([x["label"] for x in result])

#     if result_labels != expected_output:
#         #print(result_labels, expected_output)
#         string = "Case "+str(expected_output)
#         not_found = [x for x in expected_output if x not in result_labels]
#         too_much = [x for x in result_labels if x not in expected_output]
#         not_found_str = None
#         too_much_str = None
#         if len(not_found) > 0:
#             not_found_str = " failed to find "+str(not_found)
#         if len(too_much) > 0:
#             too_much_str = " found extra "+str(too_much)

#         if not_found_str and too_much_str:
#             extra = not_found_str + " and"+too_much_str
#         else:
#             if not_found_str:
#                 extra = not_found_str
#             if too_much_str:
#                 extra = too_much_str
#         string += extra
#         print(string)
#         if len(expected_output) == 1:
#             for x in expected_output:
#                 default_failed.add(x)
# # Searches the T-box index for matches for each term
