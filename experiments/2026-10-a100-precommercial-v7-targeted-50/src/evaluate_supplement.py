"""Post-freeze evaluator with isolated extension identities; original prompt/parser."""
from supplement_registry import install,memory_cap
install()
memory_cap()
from evaluate_v7 import main
if __name__=='__main__':main()
