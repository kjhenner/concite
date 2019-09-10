import jsonlines
import sys

article_path = sys.argv[1]
edge_path = sys.argv[2]

paper_ids = set()

with jsonlines.open(article_path) as reader:

