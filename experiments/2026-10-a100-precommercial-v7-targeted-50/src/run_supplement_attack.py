"""Run unchanged V7 A3 with isolated extension registration in this child only."""
from supplement_registry import install,memory_cap,verify_existing_data
install()
memory_cap()
import run_v7_attack
run_v7_attack.prepare_data=verify_existing_data
if __name__=='__main__':run_v7_attack.main()
