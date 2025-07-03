import random
import copy

#def mutate_instructions(instructions, mutation_rate=0.3):
#    """
#    Randomly mutate a list of instructions by:
#    1. Changing existing instructions
#    2. Deleting instructions
#    3. Adding new instructions
#    
#    All read functions are replaced with lambda val: None
#    
#    Args:
#        instructions: List of instruction dictionaries
#        mutation_rate: Probability of each mutation occurring (0.0 to 1.0)
#    
#    Returns:
#        A new mutated list of instructions with all functions replaced
#    """
#    # First, replace all functions with lambda val: None
#    mutated = []
#    for instr in copy.deepcopy(instructions):
#        if instr['type'] == 'r' and 'func' in instr:
#            instr['func'] = lambda val: None
#        mutated.append(instr)
#    
#    # Now perform random mutations
#    num_mutations = max(1, int(len(mutated) * mutation_rate))
#    
#    for _ in range(num_mutations):
#        mutation_type = random.choice(['change', 'delete', 'add'])
#        
#        if mutation_type == 'change' and len(mutated) > 0:
#            idx = random.randint(0, len(mutated) - 1)
#            instr = mutated[idx]
#            
#            change_what = random.choice(['type', 'addr', 'value', 'core'])
#            
#            if change_what == 'type':
#                instr['type'] = 'w' if instr['type'] == 'r' else 'r'
#                if instr['type'] == 'w' and 'value' not in instr:
#                    instr['value'] = random.randint(0, 1000)
#                elif instr['type'] == 'r':
#                    instr['func'] = lambda val: None
#                    if 'value' in instr:
#                        del instr['value']
#            
#            elif change_what == 'addr':
#                instr['addr'] = random.randint(0, 20)
#            
#            elif change_what == 'value' and instr['type'] == 'w':
#                instr['value'] = random.randint(0, 1000)
#            
#            elif change_what == 'core':
#                instr['core'] = random.randint(0, 3)
#        
#        elif mutation_type == 'delete' and len(mutated) > 1:
#            idx = random.randint(0, len(mutated) - 1)
#            del mutated[idx]
#        
#        elif mutation_type == 'add':
#            new_instr = {}
#            new_instr['type'] = random.choice(['r', 'w'])
#            new_instr['addr'] = random.randint(0, 20)
#            new_instr['core'] = random.randint(0, 1)
#            
#            if new_instr['type'] == 'r':
#                new_instr['func'] = lambda val: None
#            else:
#                new_instr['value'] = random.randint(0, 1000)
#            
#            pos = random.randint(0, len(mutated))
#            mutated.insert(pos, new_instr)
#    
#    return mutated
#
## Example usage with the original instructions
#original_instructions = [
#    {'type': 'r', 'addr': 12, 'func': '<function at 0x710fe7d8d120>', 'core': 0},
#    {'type': 'r', 'addr': 6, 'func': '<function at 0x710fe7d8d1b0>', 'core': 0},
#    {'type': 'r', 'addr': 0, 'func': '<function at 0x710fe7d8d240>', 'core': 0},
#    {'type': 'r', 'addr': 6, 'func': '<function at 0x710fe7d8d2d0>', 'core': 0},
#    {'type': 'r', 'addr': 8, 'func': '<function at 0x710fe7d8d360>', 'core': 0},
#    {'type': 'r', 'addr': 6, 'func': '<function at 0x710fe7d8d3f0>', 'core': 0},
#    {'type': 'w', 'addr': 6, 'value': 426, 'core': 0},
#    {'type': 'r', 'addr': 12, 'func': '<function at 0x710fe7da0700>', 'core': 0},
#    {'type': 'r', 'addr': 13, 'func': '<function at 0x710fe7da0790>', 'core': 0},
#    {'type': 'r', 'addr': 12, 'func': '<function at 0x710fe7da0b80>', 'core': 0},
#    {'type': 'r', 'addr': 8, 'func': '<function at 0x710fe7da0c10>', 'core': 0},
#    {'type': 'r', 'addr': 16, 'func': '<function at 0x710fe7da0ca0>', 'core': 0},
#    {'type': 'r', 'addr': 16, 'func': '<function at 0x710fe7da0670>', 'core': 0},
#    {'type': 'w', 'addr': 6, 'value': 236, 'core': 0},
#    {'type': 'r', 'addr': 20, 'func': '<function at 0x710fe7da0430>', 'core': 0},
#    {'type': 'r', 'addr': 11, 'func': '<function at 0x710fe7da03a0>', 'core': 0},
#    {'type': 'w', 'addr': 5, 'value': 978, 'core': 0},
#    {'type': 'w', 'addr': 11, 'value': 115, 'core': 0},
#    {'type': 'r', 'addr': 17, 'func': '<function at 0x710fe7da0310>', 'core': 0},
#    {'type': 'r', 'addr': 2, 'func': '<function at 0x710fe7da2680>', 'core': 0}
#]
#
## Generate mutated version
#mutated_instructions = mutate_instructions(original_instructions, mutation_rate=0.4)
#print(mutated_instructions)
#
##
### Print the result
##print("Mutated instructions with all functions as lambda val: None:")
##for i, instr in enumerate(mutated_instructions):
##    if instr['type'] == 'r':
##        # To demonstrate the function replacement, we'll show its string representation
##        print(f"{i}: {instr['type']} addr:{instr['addr']} func:{instr['func'].__name__} core:{instr['core']}")
##    else:
##        print(f"{i}: {instr['type']} addr:{instr['addr']} value:{instr['value']} core:{instr['core']}")
