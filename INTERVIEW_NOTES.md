# Notas para entrevista — churn-cost-sensitive

Documento interno. Não faz parte da documentação pública do projeto. Números de `reports/`
(Cinelume, dados simulados, 60 mil contas, teste em mai–out/2026).

---

## As 3 decisões técnicas mais importantes

### 1. Validação no tempo, com atributos "como estavam" na data de corte

**O que fiz:** uma linha por assinante ativo no dia 1º de cada mês; treino em 2024–2025, calibração
em jan–abr/2026, teste em mai–out/2026. Plano e método de pagamento são reconstruídos do histórico,
porque as colunas da tabela de assinaturas são uma foto do último dia.

**Por quê:** o mesmo assinante aparece em até 24 meses de treino. Split aleatório premia decorar
gente. Um teste apaga tudo o que aconteceu depois do corte e confere que nenhum atributo muda — e
pegou um vazamento real (taxa de abertura de e-mail).

### 2. A regra precifica o peso morto

**O que fiz:** `E = p·a·(V−O) − (1−p)·d·O − c`. O fiel que aceita o desconto custa a oferta e não
salva nada. Com d = 0 é a fórmula original, que os testes ainda fixam.

**Por quê:** a contabilidade da campanha sempre cobrou esse custo; a regra antiga não sabia dele.
Uma regra que ignora um custo que o próprio placar cobra está mal especificada.

### 3. Achar o bug de unidade antes de confiar no resultado

**O que fiz:** a partir do app 6.0, o celular grava tempo assistido em milissegundos. Corrigi
comparando com a duração de relógio de cada reprodução.

**Por quê:** sem a correção, o modelo previa 1,9% de churn contra 3,4% observado no teste e a regra
perdia para o F1. A calibração na validação estava perfeita — o problema só aparecia no tempo.

---

## 5 perguntas prováveis, com resposta

### 1. "O ganho sobre o F1 é de R$ 10.761. É real?"

Bootstrap por assinante (não por linha, porque o mesmo assinante aparece em até seis meses): IC 95%
de R$ 4.272 a R$ 16.741. Ganha nos seis meses. Mas é desigual: no Premium quase empata (+R$ 1.634),
porque o limiar do Premium (19%) fica perto do corte de F1 (15,5%); no Básico (limiar 26,5%) a
diferença é R$ 5.105. Em busca paga e indicação o F1 fica levemente à frente.

### 2. "De onde vem a taxa de aceite de 35%?"

Premissa declarada. Nenhuma oferta foi feita a assinante ativo nos dados. Na sensibilidade, a regra
fica positiva em todas as taxas; o F1 perde dinheiro em 8 de 20 cenários. Com aceite de 15% o limiar
0,5 ganha da regra (R$ 3.662 contra R$ 1.865) — está no README.

### 3. "Por que não contatar todo mundo?"

Salva 2.000 e perde R$ 3,3 milhões. A oferta (cerca de uma mensalidade) cai em 177 mil
assinantes-mês para salvar 2 mil.

### 4. "Você usou o gabarito da simulação?"

Só depois dos relatórios fechados, num módulo separado. Teste estático, checagem de caminho e hook
de auditoria impedem o treino de ler a pasta. O modelo recuperou o engajamento latente (Spearman
−0,54). Pontos cegos: fim de série e buffering. E eu errei: tratei o e-mail de reengajamento como
disparado por regra, mas era sorteado — efeito nulo, como plantado.

### 5. "Está pronto para produção?"

A regra sim; a premissa de aceite, não. O próximo passo é um teste com grupo de controle para medir
aceite e peso morto, e recalibrar o modelo todo mês com o último mês fechado.

---

## Números para ter na ponta da língua

| | |
| --- | --- |
| Dados | 59.952 contas, 37 meses, 11,2 mi reproduções, 56,2 mi eventos |
| Linhas do modelo | 605.245 (teste: 177.089 assinantes-mês) |
| Churn voluntário mensal | 3,65% |
| AUC teste / logística | 0,863 / 0,848 |
| Previsto vs observado (teste) | 3,33% / 3,36% |
| Limiar de break-even | 16% a 29% conforme o plano |
| Valor líquido: limiar 0,5 | R$ 23.182 (589 contatos) |
| Valor líquido: melhor F1 | R$ 43.856 (5.774 contatos) |
| Valor líquido: valor esperado | R$ 54.617 (3.255 contatos) |
| Deixado na mesa pelo 0,5 | R$ 31.435 |
