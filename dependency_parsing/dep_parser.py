from core import DependencyParser

def get_dependency(rows):
    a = DependencyParser()
    return a.parse_rows(rows)

# print(get_dependency([("Bahkan", "SCONJ"), ("ia", "PRON"), ("menolak", "VERB")]))