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
        "RDF_types": get_rdf_datatypes,
        "Ontologies": get_ontologies
    }
    return func_dict[type](url_endpoint)
from dataclasses import dataclass, field
from typing import List

# Base class
@dataclass
class ResultDocument:
    termIRI: str = "None"
    RDF_type: str = "None"
    label: str = "None"
    description: str = "None"
    language: str = "None"
    ontology: str = "None"

# Individual class inheriting from ResultDocument
@dataclass
class Individual(ResultDocument):
    domain: List[str] = field(default_factory=list)
    range: List[str] = field(default_factory=list)

# Class class inheriting from ResultDocument
@dataclass
class Class(ResultDocument):
    subclass: List[str] = field(default_factory=list)
    superclass: List[str] = field(default_factory=list)

# DatatypeProperty class inheriting from ResultDocument
@dataclass
class DatatypeProperty(ResultDocument):
    domain: List[str] = field(default_factory=list)
    range: List[str] = field(default_factory=list)

# ObjectProperty class inheriting from ResultDocument
@dataclass
class ObjectProperty(ResultDocument):
    domain: List[str] = field(default_factory=list)
    range: List[str] = field(default_factory=list)

# RDFType class inheriting from ResultDocument
@dataclass
class RDFType(ResultDocument):
    superclass: List[str] = field(default_factory=list)

# Ontology class
@dataclass
class Ontology:
    ontologyIRI: str = "None"
    classes: List[str] = field(default_factory=list)
    dataproperties: List[str] = field(default_factory=list)
    language: str = "None"


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
            # OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }
            # OPTIONAL { ?term dbo:name ?label . }
            # OPTIONAL { ?term dbp:name ?label . }
            # OPTIONAL { ?term rdf:ID ?label . }
            
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
        doc = DatatypeProperty()
        doc.termIRI = r["term"]["value"]

        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                doc.language = r["label"]["xml:lang"]
            doc.label = r["label"]["value"]
        if "domains" in r.keys():
            doc.domain = r["domains"]["value"].split(", ")
        if "ranges" in r.keys():
            doc.range = r["ranges"]["value"].split(", ")
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                doc.language = r["description"]["xml:lang"]
            doc.description = r["description"]["value"]
        if "ontology" in r.keys():
            doc.ontology = r["ontology"]["value"]
            if "cenguix" in doc.ontology:
                doc.ontology = doc.ontology.split("/relations")[0].split("_")[-1]
            elif "http" in doc.ontology:
                doc.ontology = doc.ontology.replace("http://", "") 
                
        
        all_data.append(doc)
        
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
            # OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }
            # OPTIONAL { ?term dbo:name ?label . }
            # OPTIONAL { ?term dbp:name ?label . }
            # OPTIONAL { ?term rdf:ID ?label . }
            
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
        doc = ObjectProperty()
        doc.termIRI = r["term"]["value"]

        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                doc.language = r["label"]["xml:lang"]
            doc.label = r["label"]["value"]
        if "domains" in r.keys():
            doc.domain = r["domains"]["value"].split(", ")
        if "ranges" in r.keys():
            doc.range = r["ranges"]["value"].split(", ")
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                doc.language = r["description"]["xml:lang"]
            doc.description = r["description"]["value"]
        if "ontology" in r.keys():
            doc.ontology = r["ontology"]["value"]
            if "cenguix" in doc.ontology:
                doc.ontology = doc.ontology.split("/relations")[0].split("_")[-1]

            elif "http" in doc.ontology:
                doc.ontology = doc.ontology.replace("http://", "") 
                
        
        all_data.append(doc)
        
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
            
            (GROUP_CONCAT(DISTINCT ?label; SEPARATOR="--||||--||||--") AS ?labels)
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
            # TODO: turn this into a BIND command: (IF(BOUND(?label), ?label, STRAFTER(STR(?term), "#")) AS ?label) 
            
            # Attempting to retrieve labels
            OPTIONAL { ?term rdfs:label ?label . }
            # OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }
            # OPTIONAL { ?term dbo:name ?label . }
            # OPTIONAL { ?term dbp:name ?label . }
            # OPTIONAL { ?term rdf:ID ?label . }

            # Attempting to retrieve description
            OPTIONAL { ?term terms:description ?description . }
            OPTIONAL { ?term rdfs:comment ?description . } 
        }
        GROUP BY ?term ?label ?description ?ontology 

    """
    
    # TODO: Get the ?labels
    # Do the check for similar basename between the term and the ontology that contains the term 
    # here instead of inside the query
    
    
    sparql = SPARQLWrapper(url_endpoint)
    sparql.setQuery(query)
    
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    all_data = []
    for r in results:
        doc = Class()
        doc.termIRI = r["term"]["value"]

        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                doc.language = r["label"]["xml:lang"]
            doc.label = r["label"]["value"]
        if "subclasses" in r.keys():
            doc.subclass = r["subclasses"]["value"].split(", ")
        if "superclasses" in r.keys():
            doc.superclass = r["superclasses"]["value"].split(", ")
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                doc.language = r["description"]["xml:lang"]
            doc.description = r["description"]["value"]
        if "ontology" in r.keys():
            doc.ontology = r["ontology"]["value"]
            if "cenguix" in doc.ontology:
                doc.ontology = doc.ontology.split("/relations")[0].split("_")[-1]
            elif "http" in doc.ontology:
                doc.ontology = doc.ontology.replace("http://", "") 
                
        all_data.append(doc)
        
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
            (GROUP_CONCAT(DISTINCT ?domainlabel; SEPARATOR=", ") AS ?domainlabels)
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
            # OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }
            # OPTIONAL { ?term dbo:name ?label . }
            # OPTIONAL { ?term dbp:name ?label . }
            # OPTIONAL { ?term rdf:ID ?label . }
            
            # Attempting to retrieve description
            OPTIONAL { ?term terms:description ?description . }
            OPTIONAL { ?term rdfs:comment ?description . }
            
            
            OPTIONAL { ?domain rdfs:label ?domainlabel . }
            OPTIONAL { ?domain skos:prefLabel ?domainlabel . }
        }
        GROUP BY ?term ?label ?class ?description ?ontology

    """
    sparql = SPARQLWrapper(url_endpoint)
    sparql.setQuery(query)
    
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    all_data = []
    for r in results:
        doc = Individual()
        doc.termIRI = r["term"]["value"]

        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                doc.language = r["label"]["xml:lang"]
            doc.label = r["label"]["value"]


        if "domainlabels" in r.keys():
            doc.domainlabels = r["domainlabels"]["value"].split(", ")
        if "ranges" in r.keys():
            doc.range = r["ranges"]["value"].split(", ")
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                doc.language = r["description"]["xml:lang"]
            doc.description = r["description"]["value"]
        if "ontology" in r.keys():
            doc.ontology = r["ontology"]["value"]
            if "cenguix" in doc.ontology:
                doc.ontology = doc.ontology.split("/relations")[0].split("_")[-1]

            elif "http" in doc.ontology:
                doc.ontology = doc.ontology.replace("http://", "") 
                
        all_data.append(doc)
        
    return all_data

def get_rdf_datatypes(url_endpoint):

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
            # OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }
            # OPTIONAL { ?term dbo:name ?label . }
            # OPTIONAL { ?term dbp:name ?label . }
            # OPTIONAL { ?term rdf:ID ?label . }

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
        doc = RDFType()
        doc.termIRI = r["term"]["value"]

        if "label" in r.keys():
            if "xml:lang" in r["label"].keys():
                doc.language = r["label"]["xml:lang"]
            doc.label = r["label"]["value"]

        if "superclasses" in r.keys():
            doc.superclass = r["superclasses"]["value"].split(", ")
        if "description" in r.keys():
            if "xml:lang" in r["description"].keys():
                doc.language = r["description"]["xml:lang"]
            doc.description = r["description"]["value"]
        if "ontology" in r.keys():
            doc.ontology = r["ontology"]["value"]
            if "cenguix" in doc.ontology:
                doc.ontology = doc.ontology.split("/relations")[0].split("_")[-1]

            elif "http" in doc.ontology:
                doc.ontology = doc.ontology.replace("http://", "") 
                
        all_data.append(doc)
        
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

def get_ontology_metadata(url_endpoint):
    # Get title,
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
            ?description ?ontologyNamespace
        WHERE {
            # Identifying the class
            ?term a owl:Ontology.

            # Attempting to retrieve the ontology URI from the class's base URI or RDF context 
            BIND(IRI(REPLACE(STR(?term), "(#|/)[^#/]*$", "")) AS ?ontologyNamespace)

            # Attempting to retrieve labels
            OPTIONAL { ?term rdfs:label ?label . }
            OPTIONAL { ?term foaf:name ?label . }
            OPTIONAL { ?term skos:prefLabel ?label . }
            OPTIONAL { ?term dc:title ?label . }
            OPTIONAL { ?term dcterms:title ?label . }

            # Attempting to retrieve description
            OPTIONAL { ?term terms:description ?description . }
            OPTIONAL { ?term rdfs:comment ?description . }
            OPTIONAL { ?term dcterms:abstract ?description . }
            
            
            
        }
        GROUP BY ?term ?label ?description ?ontologyNamespace
        
    """
    # TODO: ?s -> ontologyIRI
    # get title, description
    
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
    res = [x["ontologyFile"]["value"] for x in results["results"]["bindings"]]
    print("res", res)
    return res

def get_ontologies(url_endpoint):
 
    # Helper function to run SPARQL queries
    def run_sparql_query(query):
        sparql = SPARQLWrapper(url_endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()["results"]["bindings"]

    # Query to get details for each ontology
    def get_ontology_details(ontology_iri):
        print("Trying", ontology_iri)
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT 
                            ?dataproperty  
                            ?class
            WHERE {{
                ?dataproperty a owl:DatatypeProperty .
                ?class a owl:Class .
                
                FILTER(STRSTARTS(STR(?class), "{ontology_iri}"))
                FILTER(STRSTARTS(STR(?dataproperty), "{ontology_iri}"))
            }}

        """


        return run_sparql_query(query)

    # Extract the meaningful attributes for each ontology
    def extract_ontology_data(ontology_iri):
        #print(ontology_iri)
        ontology_details = get_ontology_details(ontology_iri)
        # for x in ontology_details:
        #     print(x)
        if not ontology_details:
            return None
        
        doc = Ontology()
        doc.ontologyIRI = ontology_iri

        # Extract label, description, version
        
        # doc.label = ontology_detail.get("label", {}).get("value", "")
        # doc.description = ontology_detail.get("description", {}).get("value", "")
        # doc.version = ontology_detail.get("version", {}).get("value", "")
        doc.classes = list(set([x.get("class", {}).get("value", "") for x in ontology_details]))
        doc.dataproperties = list(set([x.get("dataproperty", {}).get("value", "") for x in ontology_details]))
        # doc.individualsCount = int(ontology_detail.get("individualCount", {}).get("value", 0))
        print(doc.classes)
        return doc

    # Get all ontology names
    all_ontologies = get_ontology_iri(url_endpoint)

    # Embed meaningful data for each ontology
    all_ontology_data = []
    for ontology_iri in all_ontologies:
        ontology_data = extract_ontology_data(ontology_iri)
        if ontology_data:
            all_ontology_data.append(ontology_data)
    
    return all_ontology_data
# Embeds using the IRI + label
def embed_using_label(data, model):

    formatted_str = f"{data.label}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)
    
# Embeds using the IRI + description
def embed_using_desc(data, model):

    formatted_str = f"{data.description}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + domain + range
def embed_using_domain_plus_range(data, model):
   
    formatted_str = f"{data.domain} + {data.range}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + domain
def embed_using_domain(data, model):
    
    formatted_str = f"{data.domain}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + range
def embed_using_range(data, model):
    
    formatted_str = f"{data.range}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the subclass
def embed_using_subclass(data, model):
    
    formatted_str = f"{data.subclass}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the superclass
def embed_using_superclass(data, model):
    
    formatted_str = f"{data.superclass}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

# Embeds using the IRI + subclass + superclass
def embed_using_subclass_plus_superclass(data, model):
    
    formatted_str = f"{data.subclass} + {data.superclass}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

def embed_ontology_classes(data, model):
    
    formatted_str = f"{data.classes}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)

def embed_ontology_dataproperties(data, model):
    
    formatted_str = f"{data.dataproperties}"
    #print("Embedding", formatted_str)
    return model.embed_query(formatted_str)



def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
