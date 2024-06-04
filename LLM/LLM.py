from dataclassses import dataclass
from typing import lists
import json
 
from langchain.llms import HuggingFaceHub
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAchain
from langchain.prompts import promtTemplate