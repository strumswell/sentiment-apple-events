# Plan

1) The degree of Customer Engagement in the discussion of
certain Topics (issues) for each Brand
    - Data cleaning (Lower, Stemming bzw das andere, Stopwords, MD-Filter)
    - Topics identifizieren und "zählen"
        - Manuell (wie oft MacBook, ...?)
        - Topic Modelling (LDA, LSA) -> Qualität?
        - Verhältnis Topic/Summe -> Was ist am relevantesten?
    - For each Event vs. All events (Welches Thema ist am wichtigsten?)
    - Beteiligen sich Autoren eventübergreifend (Loyalität, Top-User bzw. Top-Kunden -> Wie sind die drauf? Deren Meinung hat Gewicht?)

2) The Sentiment of customer opinions on each topic about each Brand
    - Kommentare in Arrays (Topics) einsortiert
    - Sentiment-Funktion über Arrays laufen lassen (ggf. anpassen)
    - Coole Darstellung in Plots? (Topic = Plot, Comment = Punkt im Plot)
    - Sentiment von Topics über Zeitraum (Bar-Chart)
    - WordClouds (neg, pos, neut)

3) Main Thematic streams in studied comments in the selected period (Emergence, Disappearance, Peaks, Drops).
    - Pro Topic eine Linie im Plot
    - Linie verläuft über drei Jahre (Tops? Downs?)
    - Linie verläuft zur Eventzeit (1-2h) -> Ausmachen, was wann genau vorgestellt wurde
    - Linie verläuft von Pre- zu Post-Event -> Ausmachen von Hot-Topics des Events und wann
    - Beteiligung (Anzahl Kommenatare) pro Thema (vlt. Bar-Chart pro z.B. iPhone-Event)
    
4) Clusters of Brands within each Topic based on the degree of Engagement and the Polarity of customer opinions (only for those that are common to several/ all Brands)
    - Anwendbar bei uns? Sie mal fragen bei Gelegenheit.

5) General clusters of Brands.
    - Anwendbar bei uns? Sie mal fragen bei Gelegenheit.

Weitere Ideen:
    - Aktivsten Autoren aus 1.
        - Über welche Theman sprechen die? Welches Produkt ist für sie am relevantesten?
        - Was ist deren Durchschnitssscore? Haben sie einen höheren Score?
    - Score als Weightsystem für Kommentare
        - Kommentare mit höheren Score haben ein höheres Gewicht bei der Sentimentanalyse
            - Weight = 
                (score + abs(negativster_score)) / 
                sum(foreach((score + abs(negativster_score)))) // Summe aller angepassten Scores
                = 0..1
        - Thema der Kommentare über Durchschnitsscore
    - Welche Kommentare mit höchsten Score haben negatives Sentiment?
    - Welche Kommentare mit höchsten Score haben positives Sentiment?
    - Topics der Kommenatre mit niedrigstem Score
    - Similarity von Kommentaren in einem Event in einem Plot (cosine distance)
        - k-means anwenden (Qualität???)
    - Bigrams!


