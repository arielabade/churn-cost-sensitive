# Notas para entrevista — churn-cost-sensitive

Documento interno. Não faz parte da documentação pública do projeto.

---

## As 3 decisões técnicas mais importantes

### 1. Threshold por cliente, não global

**O que fiz:** resolvi "valor esperado de contatar = 0" para a probabilidade de churn. O
`value_at_risk` fica no denominador, então o threshold depende do cliente.

**Por quê:** cliente de R$110/mês vale 12 meses × 55% de margem = R$726. Cliente de R$25/mês vale
R$165. Contatar custa o mesmo para os dois. Logo, vale a pena agir sobre o caro com risco muito
menor.

**O número que fecha o argumento:** o break-even vai de 2,7% (percentil 10) a 19,6% (percentil 90).
Sete vezes de variação. Qualquer corte único está fazendo média disso.

### 2. Custo da oferta ponderado por aceitação

**O que fiz:** `offer_cost × P(churn) × taxa_de_aceitação`, não `offer_cost` para todo mundo contatado.

**Por quê:** a oferta só é paga quando aceita. O contato é pago sempre. São dois custos com naturezas
diferentes e colapsar os dois erra em direções opostas: cobrar a oferta inteira de todos superestima o
custo e faz contatar de menos; tratar a margem salva como certa superestima o ganho e faz contatar de
mais.

### 3. Calibração isotônica antes de decidir

**O que fiz:** calibrei o LightGBM antes de comparar a probabilidade com o threshold.

**Por quê:** AUC é invariante a qualquer transformação monotônica do score. Um modelo pode ordenar
perfeitamente e colocar o break-even no lugar errado. Como aqui a probabilidade é **multiplicada por
dinheiro**, ela precisa ser uma probabilidade de verdade, não um ranking. Brier 0,142.

---

## 5 perguntas prováveis, com resposta

### 1. "O ganho sobre o threshold de F1 ótimo foi de 0,8%. Isso justifica a complexidade?"

Essa é a pergunta certa e eu coloquei a ressalva no próprio README, não escondi.

Nesse dataset, com essa economia, varrer threshold para F1 pega quase todo o ganho. O resultado que eu
defendo não é contra o F1 ótimo — é contra o **0,5**, que é o que sai do `predict()` por padrão e o
que a maioria dos projetos entrega. Contra ele a diferença é £10.285, ou 21% do valor alcançável.

E tem uma razão estrutural para o gap ser pequeno aqui: a dispersão de valor entre clientes do Telco é
moderada. Num negócio com faixa de ticket mais larga, ou com oferta mais cara, o corte global quebra
mais feio. A fórmula mostra exatamente quando: quanto maior a variância de `value_at_risk`, pior o
threshold único.

Se me perguntarem "então use F1", eu respondo: F1 ótimo não te diz **quanto** você ganha, nem como o
resultado muda se a taxa de aceitação cair de 35% para 20%. O modelo econômico diz.

### 2. "De onde vem a taxa de aceitação de 35%?"

É premissa declarada. O dataset não publica, e eu não invento que publica.

É também a premissa mais frágil do projeto, e eu diria isso numa reunião antes que alguém perguntasse.
Todo business case de retenção depende dela e quase nenhum a declara.

A única forma de aprender o número real é segurar um grupo de controle e medir. Nenhum modelo de churn
substitui isso. É por isso que no README a limitação principal não é o modelo, é a falta do
experimento.

### 3. "Contatar todo mundo salva mais clientes. Por que não fazer isso?"

Salva mesmo: 146 contra 135. E destrói valor: £36.950 contra £48.621.

A conta é simples: a oferta é paga em 1.761 contatos para reter 146 pessoas. O custo de ofertar para
quem nunca ia sair come a margem de quem ia.

Essa linha está na tabela de propósito. É o contraponto que mostra que o objetivo não é maximizar
clientes salvos, é maximizar valor — e que as duas coisas apontam para lados diferentes.

### 4. "Como você tratou o TotalCharges em branco?"

São 11 linhas. Todas com `tenure = 0` — são clientes novos que ainda não foram faturados.

Preenchi com 0, não com a média. Preencher com a média inventaria onze meses de histórico de cobrança
para quem não tem nenhum, e aí `tenure` passaria a contradizer `TotalCharges` — uma inconsistência que
modelo de árvore explora alegremente e que vira ganho fantasma na validação.

Tem teste garantindo que todo cliente com tenure 0 tem TotalCharges 0.

### 5. "Esse modelo está pronto para produção?"

O modelo não, e a razão não é performance.

Faltam duas coisas. A primeira é validação temporal: o dataset é um snapshot sem coluna de data, então
validei com split aleatório estratificado. Modelo de churm em produção tem que ser validado num
período posterior ao de treino, porque o comportamento muda.

A segunda é mais séria e é conceitual: isso prevê **quem sai**, não **quem é persuadível**. Parte dos
clientes de score alto vai sair com oferta ou sem. Parte ficaria de qualquer jeito. O dinheiro está na
diferença — no uplift — e uplift não se estima com modelo de churn, se estima com experimento.

O que eu levaria para produção primeiro não seria o modelo, seria o teste controlado que mede a taxa
de aceitação real.

---

## Números para ter na ponta da língua

| | |
| --- | --- |
| Clientes | 7.043 (5.282 treino / 1.761 teste) |
| Taxa de churn | 26,5% |
| AUC / Brier | 0,848 / 0,142 |
| Break-even: p10 / mediana / p90 | 2,7% / 4,1% / 19,6% |
| Valor líquido: threshold 0,5 | £38.336 (450 contatos) |
| Valor líquido: valor esperado | £48.621 (975 contatos) |
| Valor líquido: contatar todos | £36.950 (1.761 contatos) |
| Dinheiro deixado na mesa pelo 0,5 | £10.285 (21%) |
