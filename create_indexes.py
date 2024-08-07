from sparql.Endpoint import Endpoint
from nlp.normalizer import *
# from index.whoosh_index import *
from configs import *
from index.import_index import *
import os
import shutil
from langchain_community.embeddings import SentenceTransformerEmbeddings

def createIndexes(embedding_function,embedding_method, ontology=None, search_type="classes"):
    
    if ontology:
        if not os.path.exists(f"sparql/temp/{ontology}_{search_type}"):
            os.mkdir(f"sparql/temp/{ontology}_{search_type}")
            
        if os.path.exists(f"sparql/temp/{ontology}_{search_type}/labels.obj"):
            os.remove(f"sparql/temp/{ontology}_{search_type}/labels.obj")
            print(f"Old file removed: sparql/temp/{ontology}_{search_type}/labels.obj")

        if os.path.exists(f"sparql/temp/{ontology}_{search_type}/counts.obj"):
            os.remove(f"sparql/temp/{ontology}_{search_type}/counts.obj")
            print(f"Old file removed: sparql/temp/{ontology}_{search_type}/counts.obj")

        if not os.path.exists(f"index/temp/{ontology}_{search_type}"):
            os.mkdir(f"index/temp/{ontology}_{search_type}")
        
        if os.path.exists(f"index/temp/{ontology}_{search_type}/t_box_index/"+INDEX_T_BOX.lower()):
            shutil.rmtree(f"index/temp/{ontology}_{search_type}/t_box_index/"+INDEX_T_BOX.lower())
            print(f"Old index removed: index/temp/{ontology}_{search_type}/t_box_index/"+INDEX_T_BOX.lower())

        # if os.path.exists(f"index/temp/{ontology}_{search_type}/a_box_index/"+INDEX_A_BOX.lower()):
        #     shutil.rmtree(f"index/temp/{ontology}_{search_type}/a_box_index/"+INDEX_A_BOX.lower())
        #     print(f"Old index removed: index/temp/{ontology}_{search_type}/a_box_index/"+INDEX_A_BOX.lower())
    
    
    if not ontology:
        if os.path.exists("sparql/temp/labels.obj"):
            os.remove("sparql/temp/labels.obj")
            print("Old file removed: sparql/temp/labels.obj")

        if os.path.exists("sparql/temp/counts.obj"):
            os.remove("sparql/temp/counts.obj")
            print("Old file removed: sparql/temp/counts.obj")

        if os.path.exists(f"index/temp/{search_type}/t_box_index/"+INDEX_T_BOX.lower()):
            shutil.rmtree(f"index/temp/{search_type}/t_box_index/"+INDEX_T_BOX.lower())
            print("Old index removed: index/temp/t_box_index/"+INDEX_T_BOX.lower())

        # if os.path.exists("index/temp/a_box_index/"+INDEX_A_BOX.lower()):
        #     shutil.rmtree("index/temp/a_box_index/"+INDEX_A_BOX.lower())
        #     print("Old index removed: index/temp/a_box_index/"+INDEX_A_BOX.lower())

    print("Creating T-Box endpoint")
    endpoint_t_box = Endpoint(ENDPOINT_T_BOX_URL, ontology=ontology, search_type=search_type)
    print("-----------------------------------------------------------")

    # endpoint_a_box = endpoint_t_box
    # if ENDPOINT_A_BOX_URL and ENDPOINT_T_BOX_URL != ENDPOINT_A_BOX_URL:
    #     print("Creating A-Box endpoint")
    #     endpoint_a_box = Endpoint(ENDPOINT_A_BOX_URL, ontology=ontology)
    #     print("-----------------------------------------------------------")

    print("Creating Normalizer")
    normalizer = Normalizer()
    print("-----------------------------------------------------------")

    print("Creating T-Box index")
    t_box_index = TBoxIndex(endpoint_t_box,normalizer,embedding_function,embedding_method,search_type=search_type)
    print("T-Box index created: "+str(t_box_index.exists()) + " Type: "+t_box_index.type+" Method: "+embedding_method)
    print("-----------------------------------------------------------")

    # if USE_A_BOX_INDEX:
    #     print("Creating A-Box index")
    #     a_box_index = ABoxIndex(endpoint_a_box,normalizer,embedding_function)
    #     print("A-Box index created: "+str(a_box_index.exists()) + " Type: "+a_box_index.type)
    #     print("-----------------------------------------------------------")

    print("Saving T-Box Endpoint labels cache...")
    endpoint_t_box.save_labels()
    print("T-Box Endpoint labels Saved")

    print("Saving T-Box Endpoint counts cache...")
    endpoint_t_box.save_counts()
    print("T-Box Endpoint counts Saved")

    # if USE_A_BOX_INDEX and endpoint_a_box != endpoint_t_box:
    #     print("Saving A-Box Endpoint labels cache...")
    #     endpoint_a_box.save_labels()
    #     print("A-Box Endpoint labels Saved")

    #     print("Saving A-Box Endpoint counts cache...")
    #     endpoint_a_box.save_counts()
    #     print("A-Box Endpoint counts Saved")
    # else:
    #     print("T-Box Endpoint and A-Box Endpoint are the same!")
        
    return endpoint_t_box, t_box_index, normalizer
