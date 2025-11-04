# Projeto: Classificação de Imagens

Repositório inicial para construir um modelo de classificação de imagens.

Objetivos:
- Fornecer um esqueleto de projeto em Python (PyTorch) para treinar um classificador de imagens.
- Incluir scripts, estrutura de código, dependências e testes mínimos.

Estrutura criada:

- `src/` - código fonte (modelo, dataset, treino)
- `models/` - pasta para checkpoints (criadas em tempo de treino)
- `tests/` - testes unitários (pytest)
- `requirements.txt` - dependências

Como começar (local):

1. Criar e ativar um ambiente virtual (venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instalar dependências:

```bash
pip install -r requirements.txt
```

3. Rodar um treino de exemplo (usa dados aleatórios se não houver dataset):

```bash
python src/train.py --epochs 2 --batch-size 16
```

4. Rodar testes:

```bash
pytest -q
```

Como enviar para o GitHub (exemplo):

```bash
# inicializar git local
git init
git add .
git commit -m "Initial commit: image-classification skeleton"

# criar repo remoto no GitHub via CLI (opcional):
# gh repo create <user>/<repo> --public --source=. --remote=origin --push

# ou criar repo pela UI do GitHub e depois:
# git remote add origin https://github.com/<user>/<repo>.git
# git branch -M main
# git push -u origin main
```

Licença: MIT (arquivo LICENSE incluso)

Notas:
- Este é um ponto de partida. Substitua o dataset aleatório pelo seu conjunto de imagens organizado por classes (ex: `data/train/class_x/*.jpg`).
- Ajuste o modelo e hiperparâmetros conforme necessidade.
