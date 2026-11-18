"""Validate notebook syntax and ensure checked-in outputs are cleared."""
from pathlib import Path
import ast,json
root=Path(__file__).resolve().parents[1]
notebook=json.loads((root/'notebooks/medical_record_summarization.ipynb').read_text())
assert notebook['nbformat']==4
count=0
for i,cell in enumerate(notebook['cells']):
    if cell['cell_type']=='code':
        ast.parse(''.join(cell['source']),filename=f'notebook_cell_{i}')
        assert cell['execution_count'] is None
        assert not cell['outputs']
        count+=1
print(f'Notebook checks passed: {count} code cells parse; execution counts and outputs cleared.')
