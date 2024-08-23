import os
import requests
import json
from SPARQLWrapper import SPARQLWrapper, JSON


# Gets data from the sparql endpoint
# Results come in the format [Term, Label, Description, Domain, Range, Language]
# Results are "None", when applicable
def get_data(url_endpoint, type="properties"):
    query = """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbp: <http://dbpedia.org/property/>
        PREFIX terms: <http://purl.org/dc/terms/>

        SELECT DISTINCT ?term ?label ?domain ?range ?description WHERE {
        ?term a owl:ObjectProperty.
        
        # Filtering out unwanted namespaces
        FILTER(!REGEX(STR(?term),"http://www.w3.org/2002/07/owl#","i"))
        FILTER(!REGEX(STR(?term),"http://www.w3.org/2000/01/rdf-schema#","i"))
        FILTER(!REGEX(STR(?term),"http://www.w3.org/1999/02/22-rdf-syntax-ns#","i"))
        FILTER(!REGEX(STR(?term),"http://www.w3.org/2001/XMLSchema#","i"))  
        FILTER(!REGEX(STR(?term),"http://www.ontotext.com/","i"))  
        FILTER(!REGEX(STR(?term),"nodeID","i"))
        
        # Attempting to retrieve labels
        OPTIONAL { 
            ?term rdfs:label ?label .
        }
        OPTIONAL {
            ?term foaf:name ?label .
        }
        OPTIONAL {
            ?term skos:prefLabel ?label .
        }
        OPTIONAL {
            ?term dc:title ?label .
        }
        OPTIONAL {
            ?term dcterms:title ?label .
        }
        OPTIONAL {
            ?term dbo:name ?label .
        }
        OPTIONAL {
            ?term dbp:name ?label .
        }
        
        # Attempting to retrieve domain and range
        OPTIONAL {
            ?term rdfs:domain ?domain .
        }
        OPTIONAL {
            ?term rdfs:range ?range .
        }
        
        OPTIONAL {
            ?term terms:description ?description .
        }
        OPTIONAL {
            ?term rdfs:comment ?description .
        }
        }"""
    sparql = SPARQLWrapper(url_endpoint)
    sparql.setQuery(query)
    if type == "classes":
        # change later
        pass
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    all_data = []
    for r in results:
        name = r["term"]["value"]
        lang = "None"
        label = "None"
        domain = "None"
        rang = "None"
        description = "None"
        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                lang = r["label"]["xml:lang"]
            label = r["label"]["value"]
        if "domain" in r.keys():
            domain = r["domain"]["value"]
        if "range" in r.keys():
            rang = r["range"]["value"]
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                lang = r["description"]["xml:lang"]
            description = r["description"]["value"]
        
        all_data.append([name, label, description, domain, rang, lang])
        
    return all_data

# Embeds using the IRI + label
def embed_using_label(data, model):
    term, label, description, domain, rang, language = data
    formatted_str = f"{term}: {label}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + description
def embed_using_desc(data, model):
    term, label, description, domain, rang, language = data
    formatted_str = f"{term}: {description}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + domain + range
def embed_using_domain_plus_range(data, model):
    term, label, description, domain, rang, language = data
    formatted_str = f"{term}: {domain} + {rang}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + domain
def embed_using_domain(data, model):
    term, label, description, domain, rang, language = data
    formatted_str = f"{term}: {domain}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + range
def embed_using_range(data, model):
    term, label, description, domain, rang, language = data
    formatted_str = f"{term}: {rang}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

