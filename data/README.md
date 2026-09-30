# Fonte e dicionário de dados

- **Título:** Cars - Purchase Decision Dataset.
- **Publicador:** Gabriel Santello (`gabrielsantello`).
- **Página:** https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset
- **Download público:** https://www.kaggle.com/api/v1/datasets/download/gabrielsantello/cars-purchase-decision-dataset
- **Consulta de metadados:** https://www.kaggle.com/api/v1/datasets/list?search=cars-purchase-decision-dataset
- **Licença declarada:** CC0: Public Domain. Metadados consultados em 30/09/2026, registrados em `fonte.json`.
- **Última atualização informada:** 09/07/2022. Essa é a data de atualização da publicação, não o período da coleta.
- **Arquivo utilizado localmente (não publicado):** `raw/car_data.csv`, 22.176 bytes, 1.000 registros e cinco colunas.
- **SHA-256:** `24db0b2bd9b47b15f55fb6ba0a4c59177f2ec9751eda24f3abfbbef4a012c244`.

| Coluna | Tipo | Significado e uso |
|---|---|---|
| User ID | inteiro | Identificador sem nomes ou contatos. Excluído das entradas do modelo; usado na rastreabilidade das partições. |
| Gender | texto | Categorias `Female` e `Male`. Codificação fixa 0 e 1, equivalente ao exercício. |
| Age | inteiro | Idade em anos. |
| AnnualSalary | inteiro | Renda anual na unidade da fonte. Moeda não confirmada; não converter para R$. |
| Purchased | inteiro | Alvo: 0 = não comprou; 1 = comprou. Nunca usado como entrada. |

Não foram observados nulos, duplicados completos ou identificadores repetidos. Existem 57 repetições de perfil quando o identificador é retirado. Elas são preservadas para manter a comparação com o exercício; a fonte não permite confirmar se cada repetição representa clientes independentes ou observações replicadas. Dos 200 registros de teste, 21 possuem o mesmo conjunto de atributos preditores de algum registro do treino, conforme `reports/resultados.json`. Os IDs são distintos; essa sobreposição não confirma duplicação de pessoas, mas limita a independência dos perfis avaliados.

Há 598 registros sem compra e 402 com compra. A idade varia de 18 a 63 anos e a renda de 15.000 a 152.500 unidades. A base é pública e não veio de um sistema privado da Semantix. Não se pode inferir país, empresa, amostragem, atualidade ou representatividade dos clientes a partir dos campos disponíveis.

A versão baixada coincide com a estrutura, exemplos e correlações do M40. O CSV original do curso não estava disponível para comparação integral. Nenhuma base de outro tema foi incorporada.

## Obter os dados localmente

O repositório contém somente metadados e estatísticas agregadas, conforme escolha do autor. Execute `python src/baixar_dados.py` para obter o CSV diretamente da fonte pública e conferir o hash. Dados brutos e relatórios por cliente estão excluídos no `.gitignore`.
