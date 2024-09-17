import os
import requests
import json
from SPARQLWrapper import SPARQLWrapper, JSON

# Gets data from the sparql endpoint
# Results come in the format [Term, Label, Description, Domain, Range, Language]
# Results are "None", when applicable, except Label, where it will try to create one
def get_data(url_endpoint, type="object_properties"):
    func_dict = {
        "data_properties": get_data_properties,
        "object_properties": get_object_properties,
        "classes": get_classes,
        "individuals": get_individuals,
        "RDF_types": get_rdf_types
    }
    return func_dict[type](url_endpoint)


old_namefilters = """            FILTER(!REGEX(STR(?term),"http://www.w3.org/2002/07/owl#","i"))
            FILTER(!REGEX(STR(?term),"http://www.w3.org/2000/01/rdf-schema#","i"))
            FILTER(!REGEX(STR(?term),"http://www.w3.org/1999/02/22-rdf-syntax-ns#","i"))
            FILTER(!REGEX(STR(?term),"http://www.w3.org/2001/XMLSchema#","i"))  
            FILTER(!REGEX(STR(?term),"http://www.ontotext.com/","i"))  
            FILTER(!REGEX(STR(?term),"nodeID","i"))"""

def get_data_properties(url_endpoint):
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

        SELECT DISTINCT ?term 
            (IF(BOUND(?label), ?label, STRAFTER(STR(?term), "#")) AS ?label) 
            (GROUP_CONCAT(DISTINCT ?domain; SEPARATOR=", ") AS ?domains)
            (GROUP_CONCAT(DISTINCT ?range; SEPARATOR=", ") AS ?ranges)
            ?description ?ontology
        WHERE {
            # Identifying data properties
            ?term a owl:DatatypeProperty .
            
            # Attempting to retrieve the ontology URI from the property’s base URI or RDF context
            BIND(IRI(REPLACE(STR(?term), "(#|/)[^#/]*$", "")) AS ?ontology)

            # Identifying domain and range
            OPTIONAL { ?term rdfs:domain ?domain . }
            OPTIONAL { ?term rdfs:range ?range . }

            # Attempting to retrieve labels
            OPTIONAL { ?term rdfs:label ?label . }
            OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }
            OPTIONAL { ?term dbo:name ?label . }
            OPTIONAL { ?term dbp:name ?label . }
            OPTIONAL { ?term rdf:ID ?label . }
            
            # Attempting to retrieve description
            OPTIONAL { ?term terms:description ?description . }
            OPTIONAL { ?term rdfs:comment ?description . }
            
            # Filtering out unwanted namespaces

        }
        GROUP BY ?term ?label ?description ?ontology

    """
    sparql = SPARQLWrapper(url_endpoint)
    sparql.setQuery(query)
    
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    all_data = []
    for r in results:
        name = r["term"]["value"]
        lang = "None"
        label = "None"
        domain = []
        rang = []
        description = "None"
        ontology = "None"
        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                lang = r["label"]["xml:lang"]
            label = r["label"]["value"]
        if "domains" in r.keys():
            domain = r["domains"]["value"].split(", ")
        if "ranges" in r.keys():
            rang = r["ranges"]["value"].split(", ")
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                lang = r["description"]["xml:lang"]
            description = r["description"]["value"]
        if "ontology" in r.keys():
            ontology = r["ontology"]["value"]
            if "cenguix" in ontology:
                ontology = ontology.split("/relations")[0].split("_")[-1]
            elif "http://purl.org/dc/terms" in ontology:
                ontology = "generic"
            elif "http" in ontology:
                ontology = ontology.replace("http://", "") 
                
        
        all_data.append([name, ontology, "owl_ObjectProperty", label, description, domain, rang, lang])
        
    return all_data

def get_object_properties(url_endpoint):
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

        SELECT DISTINCT ?term 
            (IF(BOUND(?label), ?label, STRAFTER(STR(?term), "#")) AS ?label) 
            (GROUP_CONCAT(DISTINCT ?domain; SEPARATOR=", ") AS ?domains)
            (GROUP_CONCAT(DISTINCT ?range; SEPARATOR=", ") AS ?ranges)
            ?description ?ontology
        WHERE {
            # Identifying object properties
            ?term a owl:ObjectProperty .
            
            # Attempting to retrieve the ontology URI from the property’s base URI or RDF context
            BIND(IRI(REPLACE(STR(?term), "(#|/)[^#/]*$", "")) AS ?ontology)

            # Identifying domain and range
            OPTIONAL { ?term rdfs:domain ?domain . }
            OPTIONAL { ?term rdfs:range ?range . }

            # Attempting to retrieve labels
            OPTIONAL { ?term rdfs:label ?label . }
            OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }
            OPTIONAL { ?term dbo:name ?label . }
            OPTIONAL { ?term dbp:name ?label . }
            OPTIONAL { ?term rdf:ID ?label . }
            
            # Attempting to retrieve description
            OPTIONAL { ?term terms:description ?description . }
            OPTIONAL { ?term rdfs:comment ?description . }
            

        }
        GROUP BY ?term ?label ?description ?ontology
    """
    sparql = SPARQLWrapper(url_endpoint)
    sparql.setQuery(query)
    
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    all_data = []
    for r in results:
        name = r["term"]["value"]
        lang = "None"
        label = "None"
        domain = []
        rang = []
        description = "None"
        ontology = "None"
        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                lang = r["label"]["xml:lang"]
            label = r["label"]["value"]
        if "domains" in r.keys():
            domain = r["domains"]["value"].split(", ")
        if "ranges" in r.keys():
            rang = r["ranges"]["value"].split(", ")
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                lang = r["description"]["xml:lang"]
            description = r["description"]["value"]
        if "ontology" in r.keys():
            ontology = r["ontology"]["value"]
            if "cenguix" in ontology:
                ontology = ontology.split("/relations")[0].split("_")[-1]
            elif "http://purl.org/dc/terms" in ontology:
                ontology = "generic"
            elif "http" in ontology:
                ontology = ontology.replace("http://", "") 
                
        
        all_data.append([name, ontology, "owl_ObjectProperty", label, description, domain, rang, lang])
        
    return all_data

def get_classes(url_endpoint):
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

        SELECT DISTINCT ?term 
            (IF(BOUND(?label), ?label, STRAFTER(STR(?term), "#")) AS ?label) 
            (GROUP_CONCAT(DISTINCT ?subclass; SEPARATOR=", ") AS ?subclasses)
            (GROUP_CONCAT(DISTINCT ?superclass; SEPARATOR=", ") AS ?superclasses)
            ?description ?ontology
        WHERE {
            # Identifying the class
            ?term a owl:Class.

            # Identifying subclasses and superclasses
            OPTIONAL { ?subclass rdfs:subClassOf ?term . }
            OPTIONAL { ?term rdfs:subClassOf ?superclass . }

            # Attempting to retrieve the ontology URI from the class's base URI or RDF context
            BIND(IRI(REPLACE(STR(?term), "(#|/)[^#/]*$", "")) AS ?ontology)

            # Filtering out unwanted namespaces


            # Attempting to retrieve labels
            OPTIONAL { ?term rdfs:label ?label . }
            OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }
            OPTIONAL { ?term dbo:name ?label . }
            OPTIONAL { ?term dbp:name ?label . }
            OPTIONAL { ?term rdf:ID ?label . }

            # Attempting to retrieve description
            OPTIONAL { ?term terms:description ?description . }
            OPTIONAL { ?term rdfs:comment ?description . }
        }
        GROUP BY ?term ?label ?description ?ontology

    """
    sparql = SPARQLWrapper(url_endpoint)
    sparql.setQuery(query)
    
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    all_data = []
    for r in results:
        name = r["term"]["value"]
        lang = "None"
        label = "None"
        subclass = []
        superclass = []
        description = "None"
        ontology = "None"
        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                lang = r["label"]["xml:lang"]
            label = r["label"]["value"]
        if "subclasses" in r.keys():
            subclass = r["subclasses"]["value"].split(", ")
        if "superclasses" in r.keys():
            superclass = r["superclasses"]["value"].split(", ")
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                lang = r["description"]["xml:lang"]
            description = r["description"]["value"]
        if "ontology" in r.keys():
            ontology = r["ontology"]["value"]
            if "cenguix" in ontology:
                ontology = ontology.split("/relations")[0].split("_")[-1]
            elif "http://purl.org/dc/terms" in ontology:
                ontology = "generic"
            elif "http" in ontology:
                ontology = ontology.replace("http://", "") 
                
        all_data.append([name, ontology, "owl_Class", label, description, subclass, superclass, lang])
        
    return all_data

def get_individuals(url_endpoint):
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

        SELECT DISTINCT ?term 
            (IF(BOUND(?label), ?label, STRAFTER(STR(?term), "#")) AS ?label) 
            ?class 
            (GROUP_CONCAT(DISTINCT ?domain; SEPARATOR=", ") AS ?domains)
            (GROUP_CONCAT(DISTINCT ?range; SEPARATOR=", ") AS ?ranges)
            ?description ?ontology
        WHERE {
            # Identifying individuals and their classes
            ?term rdf:type ?class .
            
            # Ensuring that ?term is not a class or involved in subclass relationships
            FILTER NOT EXISTS { ?term rdf:type owl:Class }
            FILTER NOT EXISTS { ?term rdf:type rdfs:Class }
            FILTER NOT EXISTS { ?term rdfs:subClassOf ?anyClass }
            FILTER NOT EXISTS { ?term owl:equivalentClass ?anyClass }
            
            # Explicitly filter out common classes or mistaken individuals
            FILTER(?class != owl:Class)
            FILTER(?class != rdfs:Class)
            
            # Identifying properties that have the individual's class as their domain or range
            OPTIONAL { ?term rdfs:domain ?domain . }
            OPTIONAL { ?term rdfs:range ?range . }

            # Attempting to retrieve the ontology URI from the individual's base URI or RDF context
            BIND(IRI(REPLACE(STR(?term), "(#|/)[^#/]*$", "")) AS ?ontology)

            # Filtering out unwanted namespaces

            
            # Attempting to retrieve labels
            OPTIONAL { ?term rdfs:label ?label . }
            OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }
            OPTIONAL { ?term dbo:name ?label . }
            OPTIONAL { ?term dbp:name ?label . }
            OPTIONAL { ?term rdf:ID ?label . }
            
            # Attempting to retrieve description
            OPTIONAL { ?term terms:description ?description . }
            OPTIONAL { ?term rdfs:comment ?description . }
        }
        GROUP BY ?term ?label ?class ?description ?ontology

    """
    sparql = SPARQLWrapper(url_endpoint)
    sparql.setQuery(query)
    
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    all_data = []
    for r in results:
        name = r["term"]["value"]
        lang = "None"
        label = "None"
        domain = []
        rang = []
        description = "None"
        ontology = "None"
        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                lang = r["label"]["xml:lang"]
            label = r["label"]["value"]
        if "domains" in r.keys():
            domain = r["domains"]["value"].split(", ")
        if "ranges" in r.keys():
            rang = r["ranges"]["value"].split(", ")
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                lang = r["description"]["xml:lang"]
            description = r["description"]["value"]
        if "ontology" in r.keys():
            ontology = r["ontology"]["value"]
            if "cenguix" in ontology:
                ontology = ontology.split("/relations")[0].split("_")[-1]
            elif "http://purl.org/dc/terms" in ontology:
                ontology = "generic"
            elif "http" in ontology:
                ontology = ontology.replace("http://", "") 
                
        
        all_data.append([name, ontology, "Individual", label, description, domain, rang, lang])
        
    return all_data

def get_rdf_types(url_endpoint):

    query = """PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX dc: <http://purl.org/dc/elements/1.1/>
            PREFIX dcterms: <http://purl.org/dc/terms/>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbp: <http://dbpedia.org/property/>
            PREFIX terms: <http://purl.org/dc/terms/>

            SELECT DISTINCT ?term 
            (IF(BOUND(?label), ?label, STRAFTER(STR(?term), "#")) AS ?label) 
            (GROUP_CONCAT(DISTINCT ?superclass; SEPARATOR=", ") AS ?superclasses)
            ?description ?ontology
            WHERE {
            # Identifying the datatype
            ?term a rdfs:Datatype.

            # Identifying superclasses (if any)
            OPTIONAL { ?term rdfs:subClassOf ?superclass . }

            # Attempting to retrieve the ontology URI from the datatype's base URI or RDF context
            BIND(IRI(REPLACE(STR(?term), "(#|/)[^#/]*$", "")) AS ?ontology)

            # Filtering out unwanted namespaces

            # Attempting to retrieve labels
            OPTIONAL { ?term rdfs:label ?label . }
            OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }
            OPTIONAL { ?term dbo:name ?label . }
            OPTIONAL { ?term dbp:name ?label . }
            OPTIONAL { ?term rdf:ID ?label . }

            # Attempting to retrieve description
            OPTIONAL { ?term terms:description ?description . }
            OPTIONAL { ?term rdfs:comment ?description . }
            }
            GROUP BY ?term ?label ?description ?ontology"""
    sparql = SPARQLWrapper(url_endpoint)
    sparql.setQuery(query)
    
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    all_data = []
    for r in results:
        name = r["term"]["value"]
        lang = "None"
        label = "None"
        subclass = []
        superclass = []
        description = "None"
        ontology = "None"
        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                lang = r["label"]["xml:lang"]
            label = r["label"]["value"]

        if "superclasses" in r.keys():
            superclass = r["superclasses"]["value"].split(", ")
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                lang = r["description"]["xml:lang"]
            description = r["description"]["value"]
        if "ontology" in r.keys():
            ontology = r["ontology"]["value"]
            if "cenguix" in ontology:
                ontology = ontology.split("/relations")[0].split("_")[-1]
            elif "http://purl.org/dc/terms" in ontology:
                ontology = "generic"
            elif "http" in ontology:
                ontology = ontology.replace("http://", "") 
                
        all_data.append([name, ontology, "RDF_datatype", label, description, "", superclass, lang])
        
    return all_data
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
            
            
    query2 = """
    SELECT DISTINCT ?ontologyFile
    WHERE {
        ?ontologyFile a <http://www.w3.org/2002/07/owl#Ontology> .
        OPTIONAL {
            ?ontologyFile <http://www.w3.org/1999/02/22-rdf-syntax-ns#about> ?identifier .
        }
        OPTIONAL {
            ?ontologyFile <http://www.w3.org/XML/1998/namespace#base> ?base .
        }
    }"""
    sparql.setQuery(query2)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for r in results["results"]["bindings"]:
        res = r["ontologyFile"]["value"]
        if not "www" in res and not ".com" in res:
            all_ontologies.append(res.replace("http://", "").replace(".owl", ""))
    return all_ontologies

# Embeds using the IRI + label
def embed_using_label(data, model):
    term, ontology, datatype, label, description, domain, rang, language = data
    formatted_str = f"{label}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + description
def embed_using_desc(data, model):
    term, ontology, datatype, label, description, domain, rang, language = data
    formatted_str = f"{description}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + domain + range
def embed_using_domain_plus_range(data, model):
    term, ontology, datatype, label, description, domain, rang, language = data
    formatted_str = f"{domain} + {rang}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + domain
def embed_using_domain(data, model):
    term, ontology, datatype, label, description, domain, rang, language = data
    formatted_str = f"{domain}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + range
def embed_using_range(data, model):
    term, ontology, datatype, label, description, domain, rang, language = data
    formatted_str = f"{rang}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + subclass
def embed_using_subclass(data, model):
    term, ontology, datatype, label, description, subclass, superclass, language = data
    formatted_str = f"{subclass}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + superclass
def embed_using_superclass(data, model):
    term, ontology, datatype, label, description, subclass, superclass, language = data
    formatted_str = f"{superclass}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + subclass + superclass
def embed_using_subclass_plus_superclass(data, model):
    term, ontology, datatype, label, description,  subclass, superclass, language = data
    formatted_str = f"{subclass} + {superclass}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
