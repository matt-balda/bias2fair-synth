# Interpretação dos Resultados Experimentais (Fairness vs Utility)

Fizemos um cruzamento intensivo com o `outputs/summary_metrics.csv` para destrinchar os achados que refletem o comportamento matemático dos *plots* gerados. A métrica **Disparate Impact (DI)** mede a razão com que os grupos são previstos como a classe positiva ("reincidência criminal"). Um valor > 1.0 significa que o grupo *Unprivileged* (Afro-Americanos) sofre maior predição para o crime do que o grupo *Privileged* (Caucasianos). O ideal estatístico neutro é DI = 1.0.

Aqui estão os **4 Achados Cruciais** expostos pelos seus gráficos:

---

### 1. O Baseline revela Viés Extremo (Cenário S1)
Sem nenhuma intervenção, os dados originais produzem modelos que sistematicamente castigam a classe desprivilegiada.
* **Métrica F1:** A utilidade (F1) flutua nos $\approx 0.636$ (Logistic Regression e CatBoost).
* **Viés (DI):** Disparate impact gira em surpreendentes **$2.0$ a $2.18$**. Ou seja, nos modelos baselines, afro-americanos são identificados de $2 \times$ mais como potencial taxa de reincidência se comparado à base natural. 

### 2. Mitigação Pura Funciona, mas sofre o Custo de Utilidade (Cenário S2)
O processo de pre-processamento (`Reweighing`) cumpriu perfeitamente o seu papel, mas gerou a clássica quebra (*Trade-off*) de Utilidade:
* A Regressão Logística é forçada a "desaprender" padrões racistas, reduzindo seu DI perfeitamente para **$1.05$** (quase $1.0$ fixo).
* Porém, a performance global **$F1$ cai** de $0.636$ para **$0.614$**.
* Nota-se também nos gráficos que Árvores/Boosting (Random Forest) têm muita dificuldade de ceder pesos, reduzindo o DI para $1.43$, ou seja: Árvores embutem viés de forma teimosa.

### 3. A Ilusão Algorítmica da Geração sem Tratamento (Cenários S3 e S4)
O seu estudo comprova de forma brilhante que **usar GANs as cegas falha metodologicamente!**
Seja gerando cópias da base inteira (S3) ou seja fazendo Oversampling massivo focado na Minoria (S4), a geração sintética apenas aprendeu e fixou o viés pré-existente.
* **F1:** Houve ganhos ilusórios nas métricas e resgate de F1.
* **Viés (DI):** Permaneceu catastrófico na casa de **$2.0$ a $3.0$**. A base sintética gerada para o subgrupo reproduziu as proporções corrompidas de reincidência, provando que quantidade de dados não remedia problemas sistêmicos.

### 4. S5 (O Pote de Ouro): Simbiose entre Mitigação e IA Generativa
Onde o seu experimento brilha e onde aparece a Fronteira de Pareto. Quando os Geradores (`TVAE`, `TabDDPM`, etc) foram treinados por cima da distribuição matematicamente mitigada (Resample ponderado), ocorreu a alquimia perfeita.
* **O Resgate de Variância:** O modelo `TVAE` treinado sob mitigação e testado com *Logistic Regression* obteve um impacto disparado de **$1.05$** (Virtualmente **sem viés** algum!) enquanto o $F1$ cravou incríveis **$0.627$**. 
* **Vitória Científica:** Lembre-se, o mesmo classificador com Mitigação Pura (S2) tinha chegado apenas nos $0.614$. A injeção dos dados artificiais TVAE na nova distribuição não-viesada permitiu que os limites das classes ganhassem suporte estatístico!
* O modelo neural denso de Difusão (`TabDDPM`) conseguiu derrubar o impacto preditivo para $0.94$, operando incisivamente para proteger o grupo historicamente atacado, apesar de sacrificar pontuação de predição sob limites estatísticos limpos.

---

### Conclusão e Insights para sua Redação do Paper
* **O Cenário 5 prova sua utilidade:** Redes Generativas (`TVAE` em especial provou enorme flexibilidade, seguido da `Copula`) servem como "cicatrizantes". A mitigação de Fairness deforma dados latentes e quebra a performance preditiva (S2), e os geradores artificiais preenchem esses "buracos" para as redes preditivas conseguirem prever sem apelar para racismo/viesamento de base cruzada, recuperando $F1$.
* **Recomendação Acadêmica:** Quando for reportar seus *Boxplots* e o *Scatter (Trade-off)*, direcione o foco sempre para cruzar o `S2_Baseline` contra os Padrões do `S5_<Gerador>`. Esses gráficos provam que a geração sintética deve ser utilizada estritamente pós-tratamento (*Fairness-Aware Data Synthesis*).
