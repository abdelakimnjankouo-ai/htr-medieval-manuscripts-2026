import os
import sys
os.environ['PYTHONUTF8'] = '1'
sys.path.insert(0, '.')

from kraken.ketos.recognition import train
from click.testing import CliRunner

files = []
for i in range(50):
    f = f'data/train_xml/ligne_{i:04d}.xml'
    if os.path.exists(f):
        files.append(os.path.abspath(f))

print(f'Fichiers trouves: {len(files)}')
print(f'Premier fichier: {files[0]}')

runner = CliRunner()
args = [
    '-i', os.path.abspath('models/McCATMuS_nfd_nofix_V1.mlmodel'),
    '-o', 'models/finetuned',
    '-N', '3',
    '-r', '0.0001',
    '-f', 'alto',
    '--resize', 'union'
] + files

print('Lancement training...')
result = runner.invoke(train, args, catch_exceptions=False)
print('STDOUT:', result.output)
if result.exception:
    import traceback
    traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
