"""Obtém a base pública localmente. O CSV não integra o repositório."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=json.loads((ROOT/'data/fonte.json').read_text(encoding='utf8'))
    with urllib.request.urlopen(source['download'],timeout=60) as response:
        archive=response.read()
    with zipfile.ZipFile(BytesIO(archive)) as z:
        matches=[name for name in z.namelist() if Path(name).name=='car_data.csv']
        if len(matches)!=1:
            raise ValueError('Conteúdo do download mudou; revise a fonte.')
        content=z.read(matches[0])
    if hashlib.sha256(content).hexdigest()!=source['sha256']:
        raise ValueError('O hash difere da base usada nesta entrega; revise a versão antes de executar.')
    target=ROOT/'data/raw/car_data.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(content)
    print('Base pública obtida localmente; hash conferido. CSV excluído da publicação por .gitignore.')

if __name__=='__main__':main()
