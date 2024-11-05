# Dockerfile for ontology search 
FROM ubuntu:20.04

# Install required packages
RUN apt-get update && \
    apt-get install -y wget git && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN mkdir -p /miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh && \
    bash /miniconda3/miniconda.sh -b -u -p /miniconda3 && \
    rm /miniconda3/miniconda.sh

# Update PATH
ENV PATH="/miniconda3/bin:${PATH}"

# Initialize Conda
RUN conda init bash && \
    conda config --set auto_activate_base false

# Create Conda environment
RUN conda create --name onto_search_env python=3.10.14 -y

# Set build argument
ARG CODE_ORIGIN=clone_github
# Copy specified directories and files to a temporary location if CODE_ORIGIN is copy_local

# create tmp directories exist for conditional copying
RUN mkdir -p /tmp/onto-search/nlp
RUN mkdir -p /tmp/onto-search/index
RUN mkdir -p /tmp/onto-search/sparql

# COPY commands with wildcards to not make the image build fail if the files do not exist
COPY --chown=root:root nlp/* /tmp/onto-search/nlp/
COPY --chown=root:root index/* /tmp/onto-search/index/
COPY --chown=root:root sparql/* /tmp/onto-search/sparql/
COPY --chown=root:root *.py /tmp/onto-search/
COPY --chown=root:root requirements.tx[t] /tmp/onto-search/requirements.txt

# Clone the Git repository, create the project root directory, or copy from context
RUN if [ "$CODE_ORIGIN" = "clone_github" ]; then \
        git clone https://github.com/JLucasFFerraz/DBpedia_doc_onto_extraction onto-search && \
        cd onto-search && \
        git checkout deployment_test; \
    elif [ "$CODE_ORIGIN" = "copy_local" ]; then \
        if [ ! -f /tmp/onto-search/requirements.txt ]; then \
            echo "Error: requirements.txt does not exist - 'copy_local' mode assumes the onto-search GitHub project in the context to build a functioning image." && exit 1; \
        fi; \
        mkdir -p /onto-search && mv /tmp/onto-search/* /onto-search/; \
    else \
        echo "Error: unknown build configuration for onto-search dockerfile: CODE_ORIGIN=$CODE_ORIGIN" && exit 1; \
    fi

# Cleanup temporary files
RUN rm -rf /tmp/onto-search

# Install Python dependencies
RUN /bin/bash -c "source activate onto_search_env && cd onto-search && pip install -r requirements.txt"

# Set working directory
WORKDIR /onto-search

# Default command
CMD ["/bin/bash"]


